#!/usr/bin/env python3
"""Frozen-prefix MOSS next-token comparison. Never generates transcript text."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

from short_windows import Guard, deadline, power
from window_matching import interrupt, save, sha256

ROOT = Path.home() / ".local/state/lalia-speaker-qualification"
BASE = ROOT / "official-api-path"
ORIGINAL = ROOT / "eos-counterfactual-20261026"
STATE = ROOT / "eos-rope-20261027"
PYTHON = Path("/opt/homebrew/opt/omlx/libexec/bin/python")
MODEL = Path.home() / ".local/state/lalia-speaker-eval-20261025/models/MOSS-Transcribe-Diarize"
PROMPT = BASE / "official-prompt.txt"
AUDIO = ROOT / "data/public/ami-ES2004a-stress-90m.wav"
PROMPT_COUNT, PREFIX_COUNT, EOS = 71687, 17127, 151645
EOS_IDS = (151643, EOS)
TOTAL = PROMPT_COUNT + PREFIX_COUNT
CHUNK = 2048
MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"


def verify(hashes):
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen file changed: {filename}")


def installed_operators():
    """Homebrew omits wheel file lists. Resolve the imported packages instead."""
    roots = {}
    for package in ("mlx", "torch", "transformers"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"Cannot locate installed {package}")
        roots[package] = Path(next(iter(spec.submodule_search_locations))).resolve()
    required = [roots["mlx"] / "lib/libmlx.dylib", roots["mlx"] / "lib/mlx.metallib",
                roots["torch"] / "lib/libtorch_cpu.dylib", roots["torch"] / "lib/libc10.dylib"]
    for module in ("mlx.core", "torch._C"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise ValueError(f"Cannot locate installed {module}")
        required.append(Path(spec.origin).resolve())
    operators = {package: sorted(path.resolve() for path in roots[package].rglob("*")
                                if path.is_file() and path.suffix in (".so", ".dylib", ".metallib"))
                 for package in ("mlx", "torch")}
    binaries = operators["mlx"] + operators["torch"]
    if not all(operators.values()) or any(not path.is_file() or path not in binaries for path in required):
        raise ValueError("Missing required MLX/Metal/Torch operator binaries")
    operators["python_sources"] = [roots["mlx"] / "nn/layers/positional_encoding.py",
                                   roots["transformers"] / "models/qwen3/modeling_qwen3.py",
                                   roots["transformers"] / "modeling_rope_utils.py"]
    if not all(path.is_file() for path in operators["python_sources"]):
        raise ValueError("Missing required RoPE Python sources")
    return operators


def prepare(state):
    """Freeze inputs and code without loading model weights or running inference."""
    if state.exists():
        raise ValueError("State already exists; no automatic rerun")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("Prepare with the isolated oMLX Python executable")
    previous = ORIGINAL / "preflight.json"
    official = BASE / "preflight.json"
    hashes = json.loads(previous.read_text())["hashes"]
    verify(hashes)
    if json.loads(official.read_text())["model_revision"][1] != MODEL_REVISION:
        raise ValueError("Unexpected model revision")
    prefix = json.loads((ORIGINAL / "original-token-prefix.json").read_text())
    validate_prefix(prefix, 151936)
    first_eos = json.loads((ORIGINAL / "first-eos.json").read_text())
    if (first_eos["sample_index"], first_eos["eos_id"], first_eos["original_response_exact_match"]) != (PREFIX_COUNT, EOS, True):
        raise ValueError("Original EOS evidence differs")
    source = Path(__file__).resolve().parent
    paths = [previous, official, PROMPT, ORIGINAL / "original-token-prefix.json",
             ORIGINAL / "original-response.json", ORIGINAL / "first-eos.json",
             ORIGINAL / "request-audit.json", source / "eos_rope.py", source / "test_eos_rope.py",
             source / "short_windows.py", source / "window_matching.py", PYTHON.resolve()]
    paths.extend(path for path in ORIGINAL.rglob("*") if path.is_file())
    # Pin all isolated runtime sources, including the shared Qwen3 attention and cache.
    paths.extend((BASE / "runtime").rglob("*.py"))
    distributions = {package: importlib.metadata.version(package)
                     for package in ("mlx", "mlx-metal", "mlx-audio", "torch", "transformers", "numpy")}
    operators = installed_operators()
    paths.extend(path for group in operators.values() for path in group)
    hashes.update({str(path): sha256(path) for path in paths})
    if sha256(PROMPT) != "5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a":
        raise ValueError("Official prompt changed")
    if hashes[str(AUDIO)] != "85a093bcdcd2d7da573ece51c4ef5ee62c26ac217171dfe0e501eef994684e68":
        raise ValueError("Public audio changed")
    protocol = {"scope": "One frozen-prefix next-token RoPE A/B; public AMI audio only",
                "model_revision": MODEL_REVISION, "python": str(PYTHON), "versions": distributions,
                "installed_operators": {name: [str(path) for path in group] for name, group in operators.items()},
                "prompt_tokens": PROMPT_COUNT, "generated_prefix_tokens": PREFIX_COUNT,
                "consumed_tokens": TOTAL, "final_position": TOTAL - 1, "chunk_tokens": CHUNK,
                "native_required_sampler_id": EOS, "eos_ids": list(EOS_IDS),
                "reference": "Torch CPU FP32 inverse frequencies/cos/sin; cast to input dtype; separately evaluated products before sum",
                "limits": {"pass_seconds": 300, "overall_seconds": 600, "sampled_footprint_bytes": 32 * 1024**3,
                           "battery_cutoff_percent": 25},
                "hashes": hashes}
    state.mkdir(mode=0o700)
    save(state / "protocol.json", protocol)
    return protocol


def validate_prefix(prefix, vocab_size, count=PREFIX_COUNT):
    if len(prefix) != count or any(type(token) is not int or not 0 <= token < vocab_size or token in EOS_IDS for token in prefix):
        raise ValueError("Invalid saved token prefix")


def assemble(prompt_ids, prompt_embeddings, prefix, embed, expected_prompt=PROMPT_COUNT, expected_prefix=PREFIX_COUNT):
    import mlx.core as mx

    validate_prefix(prefix, embed.weight.shape[0], expected_prefix)
    if prompt_ids.shape != (expected_prompt,) or prompt_embeddings.shape[0] != expected_prompt or prompt_embeddings.ndim != 2:
        raise ValueError("Prompt token/embedding length differs")
    text = embed(mx.array(prefix, dtype=mx.int32))
    embeddings = mx.concatenate((prompt_embeddings, text), axis=0)
    mx.eval(embeddings)
    return embeddings


def chunks(count, size=CHUNK):
    if count < 1 or size < 1:
        raise ValueError("Positive token/chunk counts required")
    # Preserve a single final query at the original next-token prediction position.
    for start in range(0, count - 1, size):
        yield start, min(start + size, count - 1)
    yield count - 1, count


class ReferenceRoPE:
    """Transformers Qwen3 default RoPE, with Torch CPU FP32 trigonometry."""

    def __init__(self, count, dimensions=128, base=1000000.0):
        import mlx.core as mx
        import torch

        inverse = 1 / (base ** (torch.arange(0, dimensions, 2, dtype=torch.float32) / dimensions))
        positions = torch.arange(count, dtype=torch.float32)
        angles = (inverse[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        doubled = torch.cat((angles, angles), dim=-1)
        self.cos = mx.array(doubled.cos().numpy())[:, None, :, :]
        self.sin = mx.array(doubled.sin().numpy())[:, None, :, :]
        self.dimensions = dimensions
        mx.eval(self.cos, self.sin)

    def __call__(self, value, offset=0):
        import mlx.core as mx

        end = offset + value.shape[-2]
        if value.shape[-1] != self.dimensions or offset < 0 or end > self.cos.shape[-2]:
            raise ValueError("Reference RoPE outside frozen positions/dimensions")
        cos = self.cos[:, :, offset:end, :].astype(value.dtype)
        sin = self.sin[:, :, offset:end, :].astype(value.dtype)
        half = self.dimensions // 2
        rotated = mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)
        left, right = value * cos, rotated * sin
        # Torch eager BF16 rounds each multiply. Prevent MLX fusing them into an FMA.
        mx.eval(left, right)
        return left + right


def scores(logits):
    import mlx.core as mx
    import numpy as np

    raw_dtype = str(logits.dtype)
    sampler_id = int(mx.argmax(logits - mx.logsumexp(logits, keepdims=True)).item())
    values = logits.astype(mx.float32)
    probabilities = mx.exp(values - mx.logsumexp(values))
    mx.eval(values, probabilities)
    values, probabilities = np.array(values), np.array(probabilities)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite or non-vector logits")
    order = np.argsort(-values, kind="stable")
    best = next(int(token) for token in order if token not in EOS_IDS)
    rank = lambda token: int(np.count_nonzero(values > values[token])) + 1
    return {"raw_logit_dtype": raw_dtype, "sampler_id": sampler_id, "raw_argmax_id": int(order[0]), "eos_id": EOS,
            "eos_rank": rank(EOS), "eos_logit": float(values[EOS]), "eos_probability": float(probabilities[EOS]),
            "best_non_eos_id": best, "best_non_eos_rank": rank(best), "best_non_eos_logit": float(values[best]),
            "best_non_eos_probability": float(probabilities[best]),
            "eos_minus_best_non_eos_logit": float(values[EOS] - values[best]),
            "probability_sum": float(probabilities.sum(dtype=np.float64)),
            "top": [{"id": int(token), "logit": float(values[token]), "probability": float(probabilities[token])}
                    for token in order[:10]]}


def compare(native, reference):
    result = {"native": native, "status": "native_control_not_reproduced"}
    if native["sampler_id"] == EOS:
        result.update(status="complete", reference=reference())
    return result


def token_pass(model, embeddings, state, phase, began=None):
    import mlx.core as mx

    began = time.monotonic() if began is None else began
    save(state / "progress.json", {"phase": phase, "started_monotonic": began, "consumed": 0})
    cache = model.make_cache()
    hidden = logits = None
    mx.reset_peak_memory()
    try:
        for start, end in chunks(embeddings.shape[0]):
            if any(item.offset != start for item in cache):
                raise ValueError("KV position differs from frozen prefix")
            hidden = model.model.language_model(inputs_embeds=embeddings[None, start:end, :], cache=cache)
            mx.eval(hidden, [item.state for item in cache])
            mx.clear_cache()
            save(state / "progress.json", {"phase": phase, "started_monotonic": began, "consumed": end})
            if time.monotonic() - began >= 300:
                raise TimeoutError(f"{phase} pass exceeded 300s")
        if any(item.offset != embeddings.shape[0] for item in cache):
            raise ValueError("Final cache offset differs")
        final = hidden[:, -1:, :]
        logits = model.model.language_model.embed_tokens.as_linear(final) if model.lm_head is None else model.lm_head(final)
        measured = scores(logits[0, 0])
        measured.update(wall_seconds=time.monotonic() - began, consumed_tokens=embeddings.shape[0],
                        final_position=embeddings.shape[0] - 1, mlx_peak_bytes=mx.get_peak_memory())
        return measured
    finally:
        # The caller cannot start the reference pass until this cache is gone.
        hidden = logits = None
        if "final" in locals():
            del final
        cache.clear()
        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def worker(state):
    import mlx.core as mx
    import numpy as np
    import torch
    from mlx_audio.stt.utils import load_model

    torch.set_num_threads(2)
    started = time.monotonic()
    save(state / "progress.json", {"phase": "setup", "started_monotonic": started})
    model = load_model(MODEL, strict=True)
    module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not module.is_relative_to(BASE / "runtime"):
        raise ValueError("Model did not load from frozen isolated runtime")
    if (model.config.text_config.head_dim, model.config.text_config.rope_theta) != (128, 1000000):
        raise ValueError("Unexpected RoPE configuration")
    prefix = json.loads((ORIGINAL / "original-token-prefix.json").read_text())
    response = json.loads((ORIGINAL / "original-response.json").read_text())
    if model._tokenizer.decode(prefix, skip_special_tokens=True).strip() != response["text"].strip():
        raise ValueError("Saved token prefix no longer matches original response")
    prompt_ids, prompt_embeddings, count, duration = model._prepare_generation_inputs(str(AUDIO), PROMPT.read_text())
    if count != PROMPT_COUNT or duration != 5400:
        raise ValueError("Unexpected public audio/prompt length")
    embeddings = assemble(prompt_ids, prompt_embeddings, prefix, model.model.language_model.embed_tokens)
    del prompt_embeddings
    digest = hashlib.sha256()
    for start, end in chunks(TOTAL):
        digest.update(np.array(embeddings[start:end].astype(mx.float32)).tobytes())
    save(state / "inputs.json", {"embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
         "embedding_float32_sha256": digest.hexdigest(), "setup_seconds": time.monotonic() - started,
         "prompt_token_ids_sha256": hashlib.sha256(np.array(prompt_ids).tobytes()).hexdigest(), "module": str(module)})
    native = token_pass(model, embeddings, state, "native")
    save(state / "native.json", native)

    def reference():
        began = time.monotonic()
        save(state / "progress.json", {"phase": "reference", "started_monotonic": began, "consumed": 0})
        rope = ReferenceRoPE(TOTAL)
        layers = model.model.language_model.layers
        originals = [layer.self_attn.rope for layer in layers]
        try:
            for layer in layers:
                layer.self_attn.rope = rope
            result = token_pass(model, embeddings, state, "reference", began=began)
            save(state / "reference.json", result)
            return result
        finally:
            for layer, original in zip(layers, originals):
                layer.self_attn.rope = original

    save(state / "comparison.json", compare(native, reference))


def execute(state):
    protocol = json.loads((state / "protocol.json").read_text())
    if (state / "started.json").exists():
        raise ValueError("Run already attempted; no automatic retry")
    guard = Guard(state, footprint_limit=protocol["limits"]["sampled_footprint_bytes"])
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    result = {"status": "failed"}
    try:
        verify(protocol["hashes"])
        save(state / "started.json", {"pid": os.getpid(), "unix_seconds": time.time(), "power": power()})
        guard.launch(["caffeinate", "-i", "-w", str(os.getpid())], state / "awake.log")
        environment = {**os.environ, "PYTHONPATH": str(BASE / "runtime"), "PYTHONDONTWRITEBYTECODE": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
        child = guard.launch([str(PYTHON), "-B", str(Path(__file__).resolve()), "--state", str(state), "--worker"],
                             state / "worker.log", env=environment)
        while child.poll() is None:
            guard.check()
            progress_path = state / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if progress["phase"] in ("native", "reference") and time.monotonic() - progress["started_monotonic"] >= 300:
                    raise TimeoutError(f"{progress['phase']} pass exceeded 300s")
            time.sleep(0.5)
        child.wait()
        if child.returncode:
            raise RuntimeError(f"Worker exited {child.returncode}; see worker.log")
        result.update(json.loads((state / "comparison.json").read_text()))
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        guard.close()
        result.update(wall_seconds=time.monotonic() - guard.started,
                      peak_sampled_footprint_bytes=max((s["footprint_bytes"] for s in guard.samples), default=0))
        try:
            verify(protocol["hashes"])
            result["frozen_files_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", frozen_files_unchanged=False,
                          verification_error=type(error).__name__ + ": " + str(error))
            result.setdefault("error", result["verification_error"])
            raise
        finally:
            save(state / "result.json", result)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.prepare:
        protocol = prepare(args.state)
        print(json.dumps({"state": str(args.state), "frozen_files": len(protocol["hashes"]), "inference_run": False}))
    elif args.run:
        execute(args.state)
    else:
        worker(args.state)


if __name__ == "__main__":
    main()
