#!/usr/bin/env python3
"""Validate the captured MOSS stop with one bounded, batched MLX replay."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from eos_rope import BASE, MODEL, PYTHON, NativeBF16RoPE, chunks, verify
from short_windows import Guard, deadline, power
from window_matching import interrupt, save, sha256
from trace_runtime import TRACE_STATES

SOURCE = Path(__file__).resolve().parent

PROMPT, GENERATED, EOS, WIDTH, VOCAB = 71687, 29448, 151645, 1024, 151936
TOTAL, CHUNK = PROMPT + GENERATED, 4096
EOS_IDS = (151643, EOS)
DENOMINATOR_SHA = "831fa5a3736d5380f3eea7d8413c20240f9d56f7e3c04aa92533a16dcd5227f0"
TRACE_ROOT = BASE.parent / "exact-trace-20261027"


def frozen(directory):
    raw = (directory / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (directory / "protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol changed")
    result = json.loads(raw)
    verify(result["hashes"])
    return result


def validate_trace(status, tokens, eos):
    if status["status"] != "complete" or status["reproduces_previous"] is not True:
        raise ValueError("Require a complete exact reproduction")
    observed = status["observed"]
    if (observed["output_tokens"], observed["eos_id"]) != (GENERATED, EOS):
        raise ValueError("Unexpected captured output/EOS")
    if observed != status["expected"]:
        raise ValueError("Captured output differs from previous response")
    if (len(tokens) != GENERATED + 1 or tokens[-1] != EOS
            or any(type(token) is not int or not 0 <= token < VOCAB for token in tokens)
            or any(token in EOS_IDS for token in tokens[:-1])):
        raise ValueError("Invalid captured token IDs")
    if (eos["token_id"], eos["output_tokens"], eos["yielded_tokens"], eos["argmax_id"]) != (EOS, GENERATED, GENERATED + 1, EOS):
        raise ValueError("Captured EOS pairing or sampler differs")


def prepare(state, trace, memory_gib):
    if state.exists():
        raise ValueError("State already exists. No automatic rerun.")
    trace = trace.resolve(strict=True)
    trace_root = trace.parent.parent
    if (trace.name != "trace"
            or not (trace_root == TRACE_ROOT.resolve() or trace_root.parent == TRACE_STATES.resolve())):
        raise ValueError("Require the isolated public exact trace")
    parent = frozen(trace_root)
    phase = trace.parent.name
    if parent["phases"].get(phase) is not True:
        raise ValueError("Replay requires a captured candidate phase")
    status = json.loads((trace / "status.json").read_text())
    tokens = json.loads((trace / "token-ids.json").read_text())
    eos = json.loads((trace / "eos.json").read_text())
    validate_trace(status, tokens, eos)
    if status["expected"] != parent["expected"][phase]:
        raise ValueError("Trace expectation differs from frozen request")
    verify({str(trace / "token-ids.json"): status["token_ids_sha256"],
            str(trace / "inputs.safetensors"): status["inputs"]["sha256"],
            str(trace / "combined-embeddings.safetensors"): status["combined_embeddings"]["sha256"],
            str(trace / "eos-logprobs.safetensors"): eos["artifact"]["sha256"]})
    response_path = trace.parent / "results/MOSS-Transcribe-Diarize/ami-ES2004a-stress-90m/response.json"
    response = json.loads(response_path.read_text())
    if (hashlib.sha256(response["text"].encode()).hexdigest() != status["observed"]["text_sha256"]
            or response["usage"]["input_tokens"] != PROMPT
            or response["usage"]["output_tokens"] != GENERATED):
        raise ValueError("HTTP response differs from captured reproduction")
    paths = list(trace.iterdir()) + [trace_root / "protocol.json", trace_root / "protocol.sha256"]
    paths.extend([response_path, SOURCE / "cuda/manifest.json"])
    paths.extend(SOURCE / name for name in ("exact_prefix.py", "trace_runtime.py", "test_exact_prefix.py",
                                           "cuda/score_decoder.py", "cuda/supervise.py", "cuda/smoke_numeric.py"))
    paths.extend(SOURCE / name for name in ("eos_rope.py", "short_windows.py", "window_matching.py"))
    hashes = dict(parent["hashes"])
    hashes.update({str(path): sha256(path) for path in paths if path.is_file()})
    protocol = {
        "scope": "One batched MLX replay of exact captured public inputs, conditional CUDA export",
        "trace": str(trace), "prompt_tokens": PROMPT, "generated_tokens": GENERATED,
        "consumed_tokens": TOTAL, "final_position": TOTAL - 1, "chunk_tokens": CHUNK,
        "schedule": "Whole combined prefix in 4096-token chunks, then one final query. This differs from API serial generation.",
        "required_sampler_id": EOS, "denominator_sha256": DENOMINATOR_SHA,
        "limits": {"pass_seconds": 300, "overall_seconds": 600,
                   "sampled_footprint_bytes": memory_gib * 1024**3, "battery_cutoff_percent": 25},
        "hashes": hashes,
    }
    state.mkdir(parents=True, mode=0o700)
    save(state / "protocol.json", protocol)
    (state / "protocol.sha256").write_text(sha256(state / "protocol.json") + "\n")
    return protocol


def vector_summary(values):
    import mlx.core as mx
    import numpy as np

    if values.shape != (VOCAB,):
        raise ValueError("Unexpected score vector shape")
    data = np.array(values.astype(mx.float32))
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite score vector")
    alternative = data.copy()
    alternative[list(EOS_IDS)] = -np.inf
    best = int(alternative.argmax())
    return {"argmax_id": int(mx.argmax(values).item()), "dtype": str(values.dtype),
            "eos_score": float(data[EOS]), "best_continuation_id": best,
            "best_continuation_score": float(data[best]),
            "eos_margin": float(data[EOS] - data[best]),
            "raw_sha256": hashlib.sha256(np.array(values.view(mx.uint8)).tobytes()).hexdigest()}


def compare_vectors(logits, captured):
    import mlx.core as mx

    if logits.dtype != captured.dtype:
        raise ValueError("Captured and replay score dtypes differ")
    # Match generate_step's native-dtype normalization before argmax.
    batched = logits[None]
    paired = (batched - mx.logsumexp(batched, keepdims=True)).squeeze(0)
    mx.eval(paired)
    actual, replay = vector_summary(captured), vector_summary(paired)
    if actual["argmax_id"] != EOS:
        raise ValueError("Captured score vector does not select native EOS")
    difference = paired.astype(mx.float32) - captured.astype(mx.float32)
    return {"status": "selection_reproduced" if replay["argmax_id"] == EOS else "execution_shape_mismatch",
            "captured": actual, "replay": replay, "raw_logits": vector_summary(logits),
            "logprobs_bitwise_equal": actual["raw_sha256"] == replay["raw_sha256"],
            "logprob_max_absolute_difference": float(mx.max(mx.abs(difference)).item()),
            "eos_margin_difference": replay["eos_margin"] - actual["eos_margin"]}, paired


def worker(state):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.stt.utils import load_model

    protocol = frozen(state)
    trace = Path(protocol["trace"])
    started = time.monotonic()
    save(state / "progress.json", {"phase": "setup", "started_monotonic": started, "consumed": 0})
    model = load_model(MODEL, strict=True)
    module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not module.is_relative_to(BASE / "runtime"):
        raise ValueError("Model imported outside isolated runtime")
    c = model.config.text_config
    if (c.head_dim, c.rope_theta, c.max_position_embeddings, len(model.layers)) != (128, 1000000, 131072, 28):
        raise ValueError("Unexpected MOSS architecture")
    rope = NativeBF16RoPE(131072)
    if rope.frequencies["denominator_sha256"] != DENOMINATOR_SHA:
        raise ValueError("Candidate frequencies differ")
    for layer in model.layers:
        layer.self_attn.rope = rope
    embeddings = mx.load(str(trace / "combined-embeddings.safetensors"))["inputs_embeds"]
    captured = mx.load(str(trace / "eos-logprobs.safetensors"))["logprobs"]
    if embeddings.shape != (TOTAL, WIDTH) or embeddings.dtype != mx.bfloat16:
        raise ValueError("Unexpected captured embeddings")
    digest = hashlib.sha256()
    for start, end in chunks(TOTAL, CHUNK):
        digest.update(np.array(embeddings[start:end].astype(mx.float32)).tobytes())
    status = json.loads((trace / "status.json").read_text())
    if digest.hexdigest() != status["combined_embeddings"]["float32_content_sha256"]:
        raise ValueError("Captured embedding content digest differs")
    save(state / "inputs.json", {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
                                "embedding_float32_sha256": digest.hexdigest(),
                                "source_sha256": sha256(trace / "combined-embeddings.safetensors")})
    began = time.monotonic()
    save(state / "progress.json", {"phase": "replay", "started_monotonic": began, "consumed": 0})
    cache = model.make_cache()
    try:
        for start, end in chunks(TOTAL, CHUNK):
            if any(item.offset != start for item in cache):
                raise ValueError("KV offset differs from captured prefix")
            hidden = model.model.language_model(inputs_embeds=embeddings[None, start:end], cache=cache)
            mx.eval(hidden, [item.state for item in cache])
            mx.clear_cache()
            save(state / "progress.json", {"phase": "replay", "started_monotonic": began, "consumed": end})
            if time.monotonic() - began >= 300:
                raise TimeoutError("Replay pass exceeded 300s")
        if any(item.offset != TOTAL for item in cache):
            raise ValueError("Final cache offset differs")
        final = hidden[:, -1:, :]
        logits = model.model.language_model.embed_tokens.as_linear(final) if model.lm_head is None else model.lm_head(final)
        result, paired = compare_vectors(logits[0, 0], captured)
        mx.save_safetensors(str(state / "scores.safetensors"), {"logits": logits[0, 0], "logprobs": paired})
        result.update(consumed_tokens=TOTAL, final_position=TOTAL - 1, chunk_tokens=CHUNK,
                      pass_seconds=time.monotonic() - began, frequencies=rope.frequencies,
                      scores_sha256=sha256(state / "scores.safetensors"))
        save(state / "comparison.json", result)
    finally:
        cache.clear()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def execute(state):
    protocol = frozen(state)
    if (state / "started.json").exists():
        raise ValueError("Run already attempted. No automatic retry.")
    hashes = {**protocol["hashes"], str(state / "protocol.json"): sha256(state / "protocol.json"),
              str(state / "protocol.sha256"): sha256(state / "protocol.sha256")}
    guard = Guard(state, footprint_limit=protocol["limits"]["sampled_footprint_bytes"])
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    result = {"status": "failed"}
    try:
        save(state / "started.json", {"pid": os.getpid(), "power": power(), "unix_seconds": time.time()})
        guard.launch(["caffeinate", "-i", "-w", str(os.getpid())], state / "awake.log")
        env = {**os.environ, "PYTHONPATH": str(BASE / "runtime"), "PYTHONDONTWRITEBYTECODE": "1",
               "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        child = guard.launch([str(PYTHON), "-B", str(Path(__file__).resolve()), "--state", str(state), "--worker"],
                             state / "worker.log", env=env)
        while child.poll() is None:
            guard.check()
            path = state / "progress.json"
            if path.exists():
                progress = json.loads(path.read_text())
                if progress["phase"] == "replay" and time.monotonic() - progress["started_monotonic"] >= 300:
                    raise TimeoutError("Replay pass exceeded 300s")
            time.sleep(0.5)
        child.wait()
        if child.returncode:
            raise RuntimeError(f"Worker exited {child.returncode}. See worker.log")
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
            verify(hashes)
            result["frozen_files_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", frozen_files_unchanged=False,
                          verification_error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            save(state / "result.json", result)


def export_cuda(state, output):
    import mlx.core as mx

    protocol = frozen(state)
    result = json.loads((state / "result.json").read_text())
    if result["status"] != "selection_reproduced" or result["frozen_files_unchanged"] is not True:
        raise ValueError("CUDA export requires validated MLX selection and unchanged files")
    if sha256(state / "scores.safetensors") != result["scores_sha256"]:
        raise ValueError("Replay scores changed")
    if output.exists():
        raise ValueError("CUDA bundle exists. No overwrite.")
    output.mkdir(parents=True, mode=0o700)
    inputs, probe = output / "inputs", output / "probe"
    inputs.mkdir()
    probe.mkdir()
    embeddings = mx.load(str(Path(protocol["trace"]) / "combined-embeddings.safetensors"))["inputs_embeds"]
    mx.save_safetensors(str(inputs / "combined-embeddings.safetensors"), {"embeddings": embeddings})
    manifest = json.loads((SOURCE / "cuda/manifest.json").read_text())
    status = json.loads((Path(protocol["trace"]) / "status.json").read_text())
    manifest["score_profile"] = "exact-prefix"
    manifest["embedding_float32_sha256"] = status["combined_embeddings"]["float32_content_sha256"]
    manifest["files"]["combined-embeddings.safetensors"] = sha256(inputs / "combined-embeddings.safetensors")
    for name in ("config.json", "model-00000-of-00001.safetensors"):
        if sha256(MODEL / name) != manifest["files"][name]:
            raise ValueError("Model file differs from pinned CUDA manifest")
        (inputs / name).symlink_to((MODEL / name).resolve())
    save(inputs / "manifest.json", manifest)
    for name in ("score_decoder.py", "supervise.py", "smoke_numeric.py"):
        shutil.copy2(Path(__file__).resolve().parent / "cuda" / name, probe / name)
    save(output / "source-hashes.json", {p.name: sha256(p) for p in probe.iterdir()})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--memory-gib", type=int, choices=(32, 64), default=64)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--export-cuda", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.prepare:
        if args.trace is None:
            parser.error("--prepare requires --trace")
        result = prepare(args.state, args.trace, args.memory_gib)
        print(json.dumps({"frozen_files": len(result["hashes"]), "inference_run": False}))
    elif args.run:
        execute(args.state)
    elif args.worker:
        worker(args.state)
    elif args.export_cuda:
        export_cuda(args.state, args.export_cuda)
    else:
        parser.error("Choose --prepare, --run or --export-cuda")


if __name__ == "__main__":
    main()
