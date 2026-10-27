#!/usr/bin/env python3
"""One conditional next-token score with the official Qwen3 decoder."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers.initialization import no_init_weights
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

GIB = 1024**3
TOTAL = 88814
CHUNK = 2048
EOS = 151645
R7_ROPE_SHA256 = "fb39dde72a24bd6b8b3573c5e6375d3f163568c8649e0e97f9fd27ef52f0a493"
PREFIX = "model.language_model."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def chunks(total=TOTAL, size=CHUNK):
    return [(start, min(start + size, total - 1)) for start in range(0, total - 1, size)] + [(total - 1, total)]


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def empty_decoder(config):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with no_init_weights():
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(previous)
    return model


def load_decoder(config, checkpoint):
    model = empty_decoder(config)
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        weights = {key.removeprefix(PREFIX): source.get_tensor(key)
                   for key in source.keys() if key.startswith(PREFIX)}
    if any(t.dtype != torch.bfloat16 for t in weights.values()):
        raise ValueError("Expected BF16 decoder tensors")
    model.load_state_dict(weights, strict=True, assign=True)
    # Match the pinned official ModelRunner._ensure_loaded, including RoPE buffers.
    return model.to(dtype=torch.bfloat16).eval()


def restore_fp32_rope(model):
    rotary = model.rotary_emb
    if rotary.rope_type != "default" or model.config.rope_parameters["rope_type"] != "default":
        raise ValueError("FP32 frequency control requires default RoPE")
    if rotary.inv_freq.dtype != torch.bfloat16 or tensor_sha256(rotary.inv_freq) != R7_ROPE_SHA256:
        raise ValueError("Original rotary buffer differs from R7")
    with torch.device("cpu"):
        restored, scaling = rotary.compute_default_rope_parameters(model.config)
    if restored.dtype != torch.float32 or restored.device.type != "cpu":
        raise ValueError("Expected original FP32 frequencies computed on CPU")
    if scaling != rotary.attention_scaling or scaling != 1.0:
        raise ValueError("RoPE attention scaling changed")
    if tensor_sha256(restored.bfloat16()) != R7_ROPE_SHA256:
        raise ValueError("Restored frequencies do not round back to R7")
    if torch.equal(restored, rotary.inv_freq.cpu().float()):
        raise ValueError("Frequency restoration only upcast the rounded buffer")
    # Default RoPE uses inv_freq; leave the unused original_inv_freq buffer intact.
    rotary.inv_freq = restored.to(rotary.inv_freq.device)


def score(model, embeddings, *, report=lambda row: None, chunk_size=CHUNK):
    cache = None
    began = time.monotonic()
    device = model.embed_tokens.weight.device
    for start, end in chunks(embeddings.shape[0], chunk_size):
        if (cache.get_seq_length() if cache else 0) != start:
            raise ValueError("Unexpected cache position")
        output = model(
            inputs_embeds=embeddings[None, start:end].to(device),
            position_ids=torch.arange(start, end, device=device)[None],
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
        if any(cache.get_seq_length(i) != end for i in range(model.config.num_hidden_layers)):
            raise ValueError("Cache layer lengths disagree")
        if end == embeddings.shape[0]:
            logits = torch.nn.functional.linear(output.last_hidden_state[:, -1], model.embed_tokens.weight)
        del output
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        report({"consumed": end, "seconds": time.monotonic() - began})
    if logits.dtype != torch.bfloat16:
        raise ValueError("Expected native BF16 logits")
    return logits[0], cache


def summary(logits):
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite final logits")
    raw = logits.float().cpu()
    best = int(raw.argmax())
    alternative = raw.clone()
    alternative[EOS] = -torch.inf
    non_eos = int(alternative.argmax())
    top = torch.argsort(raw, descending=True, stable=True)[:10]
    return {"argmax": best, "eos": float(raw[EOS]), "best_non_eos_id": non_eos,
            "token_42474": float(raw[42474]), "eos_minus_token_42474": float(raw[EOS] - raw[42474]),
            "best_non_eos": float(raw[non_eos]), "eos_margin": float(raw[EOS] - raw[non_eos]),
            "ties_at_max": int((raw == raw.max()).sum()), "logits_dtype": str(logits.dtype),
            "raw_bf16_sha256": tensor_sha256(logits),
            "top10": [{"id": int(i), "logit": float(raw[i])} for i in top]}


def fused_attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])


def verify_runtime():
    from packaging.requirements import Requirement
    if importlib.metadata.version("transformers") != "5.17.0":
        raise ValueError("Expected Transformers 5.17.0")
    dependencies = {}
    for package in ("transformers", "safetensors"):
        for raw in importlib.metadata.requires(package) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            actual = importlib.metadata.version(requirement.name)
            if actual not in requirement.specifier:
                raise ValueError(f"Unsatisfied probe dependency: {raw}, installed {actual}")
            dependencies[requirement.name] = actual
    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise ValueError("The standalone probe must not import vLLM")
    return dependencies


def cuda_smoke(stage="tiny"):
    from smoke_numeric import run_smoke
    return run_smoke(stage)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--cuda-shapes", action="store_true")
    parser.add_argument("--fp32-rope", action="store_true")
    args = parser.parse_args()
    if args.fp32_rope and (args.cuda_smoke or args.cuda_shapes):
        parser.error("--fp32-rope requires scoring")
    torch.set_num_threads(2)
    dependencies = verify_runtime()
    if args.cuda_smoke or args.cuda_shapes:
        stage = "shapes" if args.cuda_shapes else "tiny"
        print(json.dumps({**cuda_smoke(stage), "dependencies": dependencies}))
        return
    if args.inputs is None or args.out is None:
        parser.error("--inputs and --out are required for scoring")
    if args.out.exists():
        raise ValueError("Output exists. No automatic rerun")
    if available_bytes() < 32 * GIB:
        raise MemoryError("Need at least 32 GiB MemAvailable before scoring")
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(args.inputs / name) != digest:
            raise ValueError(f"Input hash mismatch: {name}")
    import transformers
    root = Path(transformers.__file__).parent
    for name, digest in manifest["transformers_sources"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Transformers source mismatch: {name}")
    args.out.mkdir(mode=0o700)
    torch.cuda.set_per_process_memory_fraction(24 * GIB / torch.cuda.get_device_properties(0).total_memory)
    config = Qwen3Config(**json.loads((args.inputs / "config.json").read_text())["text_config"])
    config._attn_implementation = "sdpa"
    model = load_decoder(config, args.inputs / "model-00000-of-00001.safetensors")
    original_rope_dtype = str(model.rotary_emb.inv_freq.dtype)
    original_rope_sha256 = tensor_sha256(model.rotary_emb.inv_freq)
    if args.fp32_rope:
        restore_fp32_rope(model)
    model = model.to("cuda")
    with safe_open(args.inputs / "combined-embeddings.safetensors", framework="pt", device="cpu") as source:
        embeddings = source.get_tensor("embeddings")
    if embeddings.shape != (TOTAL, 1024) or embeddings.dtype != torch.bfloat16:
        raise ValueError("Unexpected saved embeddings")
    digest = hashlib.sha256()
    for start in range(0, TOTAL, CHUNK):
        digest.update(embeddings[start:start + CHUNK].float().numpy().tobytes())
    if digest.hexdigest() != manifest["embedding_float32_sha256"]:
        raise ValueError("Embedding content digest differs")
    metadata = {"probe_sha256": sha256(__file__), "dependencies": dependencies,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "torch_git": torch.version.git_version, "compiled_arches": torch.cuda.get_arch_list(),
                "precision_settings": {
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    "math_sdp_reduced_precision": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()},
                "rope_dtype": str(model.rotary_emb.inv_freq.dtype),
                "rope_raw_sha256": tensor_sha256(model.rotary_emb.inv_freq),
                "fp32_rope": args.fp32_rope, "rope_original_dtype": original_rope_dtype,
                "rope_original_sha256": original_rope_sha256,
                "rope_attention_scaling": model.rotary_emb.attention_scaling,
                "model_revision": manifest["model_revision"], "chunk": CHUNK,
                "consumed": TOTAL, "final_position": TOTAL - 1,
                "attn_implementation": config._attn_implementation, "math_sdpa_enabled": False,
                "loader": "Pinned official ModelRunner BF16 cast, decoder only"}
    (args.out / "started.json").write_text(json.dumps(metadata, indent=2) + "\n")
    progress = []
    def report(row):
        row.update(available_bytes=available_bytes(), cuda_allocated=torch.cuda.memory_allocated(),
                   cuda_reserved=torch.cuda.memory_reserved(), cuda_peak=torch.cuda.max_memory_allocated())
        progress.append(row)
        (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        if row["available_bytes"] < 8 * GIB:
            raise MemoryError("MemAvailable below 8 GiB")
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16), fused_attention():
        logits, cache = score(model, embeddings, report=report)
    logits_path = args.out / "logits.safetensors"
    save_file({"logits": logits.detach().cpu().contiguous()}, logits_path)
    result = {**metadata, **summary(logits), "seconds": progress[-1]["seconds"],
              "logits_file": logits_path.name, "logits_file_sha256": sha256(logits_path),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "cache_tokens": cache.get_seq_length()}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
