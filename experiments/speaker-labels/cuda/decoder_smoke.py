"""One SDPA control for the failed fused CPU/CUDA decoder parity check."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys

import torch
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config

import score_decoder as probe
import smoke_numeric as smoke

ATOL, RTOL = 0.002, 0.02
LENGTH, CHUNK, OFFSET = 65, 32, 88814 - 65


def fixture():
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=128, hidden_size=1024, intermediate_size=3072,
                        num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=8,
                        head_dim=128, max_position_embeddings=131072, rope_theta=1000000,
                        rms_norm_eps=1e-6, tie_word_embeddings=True, pad_token_id=0,
                        attention_bias=False, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = probe.Qwen3Model(config).to(dtype=torch.bfloat16).eval()
    embeddings = torch.randn(LENGTH, 1024, dtype=torch.float32).bfloat16()
    return model, embeddings


def score(model, embeddings):
    cache = None
    device = model.embed_tokens.weight.device
    for start, end in probe.chunks(len(embeddings), CHUNK):
        if (cache.get_seq_length() if cache else 0) != start:
            raise ValueError("Unexpected fresh-cache sequence length")
        output = model(inputs_embeds=embeddings[None, start:end].to(device),
                       position_ids=torch.arange(OFFSET + start, OFFSET + end, device=device)[None],
                       past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        if any(cache.get_seq_length(i) != end for i in range(model.config.num_hidden_layers)):
            raise ValueError("Decoder cache layers disagree")
    logits = torch.nn.functional.linear(output.last_hidden_state[:, -1], model.embed_tokens.weight)[0]
    if logits.dtype != torch.bfloat16:
        raise ValueError("Expected native BF16 decoder logits")
    return logits, cache.get_seq_length()


def metrics(actual, expected):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Wrong shape or nonfinite decoder logits")
    if expected.abs().max() < 0.05:
        raise ValueError("Degenerate decoder fixture")
    difference = (actual - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    stats = {"max_abs": difference.max().item(), "rms": difference.square().mean().sqrt().item(),
             "max_tolerance_ratio": (difference / allowed).max().item(),
             "outside_tolerance": int((difference > allowed).sum())}
    return stats


def compare(actual, expected):
    stats = metrics(actual, expected)
    if stats["outside_tolerance"]:
        raise ValueError(f"CPU/CUDA decoder mismatch: {stats}")
    return stats


def weights_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(probe.tensor_sha256(value).encode())
    return digest.hexdigest()


def require_outer_product(operators):
    from torch._native import registry
    from torch._native.ops.bmm_outer_product import triton_impl
    # Torch 2.14.1 names opaque profiler nodes by registry ID, not function name.
    nodes = [node for node in registry._graphs.get(("bmm", "CUDA"), [])
             if node.active and node.dsl_name == "triton"
             and node.impl_fn is triton_impl._bmm_outer_product_impl
             and node.cond_fn is triton_impl._bmm_outer_product_cond]
    if len(nodes) != 1:
        raise ValueError("Expected one active CUDA Triton outer-product registration")
    name = f"_native::{nodes[0].node_id}"
    if name not in operators:
        raise ValueError(f"Triton RoPE outer-product path {name} was not exercised: {operators}")
    return name


def finish_control(directory, logits, metadata):
    tensors = {name: value.detach().cpu().contiguous() for name, value in logits.items()}
    path = directory / "decoder-logits.safetensors"
    save_file(tensors, path)
    result = {"type": "decoder-control-result", "control_qualified": False,
              "original_fused_parity_gate": "FAILED", **metadata,
              "logit_hashes": {name: probe.tensor_sha256(value) for name, value in tensors.items()},
              "logit_artifact": path.name, "logit_artifact_sha256": probe.sha256(path)}
    record = directory / "decoder-control.json"
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pairs = {
        "fused_vs_cpu": metrics(tensors["cuda_fused"], tensors["cpu_math"]),
        "math_vs_cpu": metrics(tensors["cuda_math"], tensors["cpu_math"]),
        "fused_vs_math": metrics(tensors["cuda_fused"], tensors["cuda_math"]),
    }
    fused = pairs["fused_vs_cpu"]
    reproduced = (fused["outside_tolerance"] == 2 and fused["max_abs"] == 0.0078125
                  and math.isclose(fused["rms"], 0.0027748744469136, rel_tol=1e-6, abs_tol=1e-12))
    result.update(pairwise=pairs, fused_failure_reproduced=reproduced,
                  cuda_math_matches_cpu=pairs["math_vs_cpu"]["outside_tolerance"] == 0)
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Evidence is durable before any backend or numerical qualification assertion.
    outer = {name: require_outer_product(result["operators"][name]) for name in ("cuda_fused", "cuda_math")}
    math_op = "aten::_scaled_dot_product_attention_math"
    fused_ops = {"aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
                 "aten::_scaled_dot_product_cudnn_attention"}
    if (math_op not in result["operators"]["cuda_math"]
            or fused_ops.intersection(result["operators"]["cuda_math"])
            or math_op in result["operators"]["cuda_fused"]
            or not fused_ops.intersection(result["operators"]["cuda_fused"])):
        raise ValueError("SDPA control did not use the requested backend paths")
    if any(length != LENGTH for length in result["cache_lengths"].values()):
        raise ValueError("Incomplete decoder control")
    if result["available_bytes_after"] < 8 * 1024**3:
        raise MemoryError("MemAvailable below 8 GiB after decoder control")
    if not reproduced or not result["cuda_math_matches_cpu"]:
        raise ValueError(f"SDPA control did not qualify: {pairs}")
    result.update(control_qualified=True, outer_product_operators=outer,
                  inference_scope="Runtime-specific fused diagnostic; original fused parity gate remains FAILED")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def worker():
    import triton
    torch.set_num_threads(2)
    if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0" or triton.__version__ != "3.8.0":
        raise ValueError("Unexpected Torch/CUDA/Triton runtime")
    initial_available = smoke.require_available(12)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("Expected Spark GB10 capability 12.1")
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model, embeddings = fixture()
    if model.rotary_emb.inv_freq.dtype != torch.bfloat16:
        raise ValueError("Official whole-model cast did not retain BF16 rotary buffer")
    state_hash = weights_sha256(model)
    rotary_hash = probe.tensor_sha256(model.rotary_emb.inv_freq)
    precision = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                 "math_sdp_reduced_precision": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}
    logits, lengths, operators = {}, {}, {}
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            logits["cpu_math"], lengths["cpu_math"] = score(model, embeddings)
    operators["cpu_math"] = sorted(event.key for event in profile.key_averages()
                                   if "bmm" in event.key or "scaled_dot_product" in event.key)
    gpu = copy.deepcopy(model).cuda()
    if weights_sha256(gpu) != state_hash or probe.tensor_sha256(gpu.rotary_emb.inv_freq) != rotary_hash:
        raise ValueError("CPU/CUDA weights or rotary buffers differ")
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel([
            SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            logits["cuda_fused"], lengths["cuda_fused"] = score(gpu, embeddings)
            torch.cuda.synchronize()
    operators["cuda_fused"] = sorted(event.key for event in profile.key_averages()
                                     if "bmm" in event.key or "scaled_dot_product" in event.key)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel([SDPBackend.MATH]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            logits["cuda_math"], lengths["cuda_math"] = score(gpu, embeddings)
            torch.cuda.synchronize()
    operators["cuda_math"] = sorted(event.key for event in profile.key_averages()
                                    if "bmm" in event.key or "scaled_dot_product" in event.key)
    metadata = {"seed": 17,
              "config": model.config.to_dict(), "length": LENGTH, "chunk": CHUNK,
              "first_position": OFFSET, "last_position": OFFSET + LENGTH - 1,
              "cache_lengths": lengths,
              "weights_sha256": state_hash, "embedding_sha256": probe.tensor_sha256(embeddings),
              "rotary_dtype": str(model.rotary_emb.inv_freq.dtype), "rotary_sha256": rotary_hash,
              "cuda_autocast_dtype": "torch.bfloat16",
              "atol": ATOL, "rtol": RTOL, "operators": operators, "precision_settings": precision,
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "triton": triton.__version__, "cuda": torch.version.cuda,
              "compiled_arches": torch.cuda.get_arch_list(), "device": torch.cuda.get_device_name(),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "initial_available_bytes": initial_available,
              "available_bytes_after": probe.available_bytes()}
    return finish_control(Path("/results"), logits, metadata)


def run(log=Path("/results/decoder-worker.log")):
    def interrupted(signum, frame):
        raise InterruptedError(f"Decoder smoke received signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with log.open("x") as output:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                       stdout=output, stderr=subprocess.STDOUT, timeout=90)
        smoke.reject_fatal_diagnostics(log)
        if completed.returncode:
            raise RuntimeError(f"Decoder worker exited {completed.returncode}; see {log}")
        results = [json.loads(line) for line in log.read_text().splitlines()
                   if line.startswith('{"type": "decoder-control-result"')]
        if (len(results) != 1 or results[0].get("control_qualified") is not True
                or results[0].get("original_fused_parity_gate") != "FAILED"):
            raise ValueError("Missing or unqualified decoder control")
        return results[0]
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    print(json.dumps(worker() if parser.parse_args().worker else run(), allow_nan=False))
