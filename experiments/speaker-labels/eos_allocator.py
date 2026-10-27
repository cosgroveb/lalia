#!/usr/bin/env python3
"""Native-only MOSS allocator comparison over eight identical prefill chunks."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import eos_rope as frozen
from short_windows import Guard, power
from window_matching import footprint, interrupt, save, sha256

STATE = frozen.ROOT / "eos-allocator-20261027"
CONSUMED = 16384
LIMIT = 32 * 1024**3
INPUT_KEYS = ("embedding_shape", "embedding_dtype", "embedding_float32_sha256", "prompt_token_ids_sha256")


def historical_hashes(state):
    hashes = json.loads((state / "protocol.json").read_text())["hashes"].copy()
    snapshot = state / "source-snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name, source in manifest.items():
        if source["in_frozen_protocol"]:
            if hashes.pop(source["source"]) != source["sha256"]:
                raise ValueError("Archived source disagrees with measured protocol")
            hashes[str(snapshot / name)] = source["sha256"]
    return hashes


def prepare(state):
    if state.exists():
        raise ValueError("State already exists; no automatic rerun")
    previous = json.loads((frozen.STATE / "protocol.json").read_text())
    historical = historical_hashes(frozen.STATE)
    frozen.verify(historical)
    source = Path(__file__).resolve().parent
    paths = [source / name for name in ("eos_allocator.py", "test_eos_allocator.py", "eos_rope.py",
                                       "test_eos_rope.py", "short_windows.py", "window_matching.py")]
    paths.extend(path for path in frozen.STATE.rglob("*") if path.is_file())
    hashes = {**historical, **{str(path): sha256(path) for path in paths}}
    protocol = {"scope": "Fresh native workers A/B, only per-chunk allocator clearing differs",
                "model_revision": previous["model_revision"], "versions": previous["versions"],
                "installed_operators": previous["installed_operators"],
                "consumed_tokens": CONSUMED, "chunk_tokens": frozen.CHUNK,
                "policies": {"A": "No per-chunk clear", "B": "mx.clear_cache after every evaluated chunk"},
                "terminal_clear": "Both workers clear after final hidden digest; no subsequent model computation",
                "limits": {"worker_seconds": 30, "overall_seconds": 60, "sampled_footprint_bytes": LIMIT,
                           "battery_cutoff_percent": 25}, "hashes": hashes}
    state.mkdir(mode=0o700)
    save(state / "protocol.json", protocol)
    return protocol


def chunks():
    return [(start, start + frozen.CHUNK) for start in range(0, CONSUMED, frozen.CHUNK)]


def same_inputs(left, right):
    if any(left[key] != right[key] for key in INPUT_KEYS):
        raise ValueError("Frozen input embeddings or prompt IDs differ")


def tensor_digest(value):
    import mlx.core as mx
    import numpy as np

    digest = hashlib.sha256()
    for start in range(0, value.shape[0], frozen.CHUNK):
        digest.update(np.array(value[start:start + frozen.CHUNK].astype(mx.float32)).tobytes())
    return digest.hexdigest()


def kv_bytes(cache):
    logical = allocated = 0
    for item in cache:
        for value in (item.keys, item.values):
            if value is not None:
                allocated += value.nbytes
                logical += value.nbytes * item.offset // value.shape[-2]
    return {"kv_logical_bytes": logical, "kv_allocated_bytes": allocated}


def sample(cache, event, consumed, began):
    import mlx.core as mx

    mx.synchronize()
    return {"event": event, "consumed_tokens": consumed, "elapsed_seconds": time.monotonic() - began,
            "active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(), "physical_footprint_bytes": footprint(os.getpid()),
            **kv_bytes(cache)}


def prefix_pass(model, embeddings, directory, clear_each_chunk):
    import mlx.core as mx

    began = time.monotonic()
    cache, measurements, hidden = model.make_cache(), [], None

    def record(event, consumed):
        row = sample(cache, event, consumed, began)
        measurements.append(row)
        save(directory / "measurements.json", measurements)
        if row["physical_footprint_bytes"] > LIMIT:
            raise MemoryError("Worker physical footprint exceeded 32GiB")

    try:
        record("before_prefix", 0)
        for start, end in chunks():
            if any(item.offset != start for item in cache):
                raise ValueError("Unexpected KV offset")
            hidden = model.model.language_model(inputs_embeds=embeddings[None, start:end, :], cache=cache)
            mx.eval(hidden, [item.state for item in cache])
            record("after_chunk", end)
            if clear_each_chunk:
                mx.clear_cache()
            record("after_policy", end)
        if any(item.offset != CONSUMED for item in cache):
            raise ValueError("Final KV offset differs")
        hidden_digest = tensor_digest(hidden)
        prefix_seconds = time.monotonic() - began
        record("before_terminal_clear", CONSUMED)
        mx.clear_cache()
        record("after_terminal_clear", CONSUMED)
        return {"final_hidden_float32_sha256": hidden_digest, "final_hidden_shape": list(hidden.shape),
                "final_hidden_dtype": str(hidden.dtype), "prefix_seconds": prefix_seconds,
                "consumed_tokens": CONSUMED, "measurements": measurements}
    finally:
        hidden = None
        cache.clear()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def worker(state, policy):
    import mlx.core as mx
    import numpy as np
    import torch
    from mlx_audio.stt.utils import load_model

    torch.set_num_threads(2)
    began = time.monotonic()
    directory = state / policy
    mx.reset_peak_memory()
    model = load_model(frozen.MODEL, strict=True)
    module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not module.is_relative_to(frozen.BASE / "runtime"):
        raise ValueError("Model did not load from frozen isolated runtime")
    prefix = json.loads((frozen.ORIGINAL / "original-token-prefix.json").read_text())
    response = json.loads((frozen.ORIGINAL / "original-response.json").read_text())
    if model._tokenizer.decode(prefix, skip_special_tokens=True).strip() != response["text"].strip():
        raise ValueError("Saved token prefix differs")
    prompt_ids, prompt_embeddings, count, duration = model._prepare_generation_inputs(str(frozen.AUDIO), frozen.PROMPT.read_text())
    if count != frozen.PROMPT_COUNT or duration != 5400:
        raise ValueError("Public audio/prompt length differs")
    embeddings = frozen.assemble(prompt_ids, prompt_embeddings, prefix, model.model.language_model.embed_tokens)
    del prompt_embeddings
    inputs = {"embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
              "embedding_float32_sha256": tensor_digest(embeddings),
              "prompt_token_ids_sha256": hashlib.sha256(np.array(prompt_ids).tobytes()).hexdigest()}
    same_inputs(inputs, json.loads((frozen.STATE / "inputs.json").read_text()))
    if policy == "B":
        same_inputs(inputs, json.loads((state / "A/inputs.json").read_text()))
    save(directory / "inputs.json", inputs)
    setup_seconds = time.monotonic() - began
    result = prefix_pass(model, embeddings, directory, clear_each_chunk=policy == "B")
    result.update(policy=policy, setup_seconds=setup_seconds, inputs=inputs)
    if policy == "A":
        # Archive only after measurements and cache release, outside the measured prefix.
        output = state / "combined-embeddings.safetensors"
        mx.save_safetensors(str(output), {"embeddings": embeddings})
        restored = mx.load(str(output))["embeddings"]
        if restored.shape != embeddings.shape or restored.dtype != embeddings.dtype or tensor_digest(restored) != inputs["embedding_float32_sha256"]:
            raise ValueError("Saved embeddings failed round-trip validation")
        result["embedding_artifact"] = {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size}
        del restored
    result["worker_seconds"] = time.monotonic() - began
    save(directory / "result.json", result)


def compare(left, right):
    same_inputs(left["inputs"], right["inputs"])
    if left["final_hidden_float32_sha256"] != right["final_hidden_float32_sha256"]:
        raise ValueError("Final hidden states differ")
    return {"status": "complete", "inputs_equal": True, "final_hidden_equal": True, "A": left, "B": right}


def deadline(*_):
    raise TimeoutError("Overall 60s deadline")


def execute(state):
    protocol = json.loads((state / "protocol.json").read_text())
    if (state / "started.json").exists():
        raise ValueError("Run already attempted; no automatic retry")
    guard = Guard(state, footprint_limit=LIMIT)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    result = {"status": "failed"}
    try:
        frozen.verify(protocol["hashes"])
        save(state / "started.json", {"pid": os.getpid(), "unix_seconds": time.time(), "power": power()})
        guard.launch(["caffeinate", "-i", "-w", str(os.getpid())], state / "awake.log")
        environment = {**os.environ, "PYTHONPATH": str(frozen.BASE / "runtime"), "PYTHONDONTWRITEBYTECODE": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
        results = []
        for policy in ("A", "B"):
            directory = state / policy
            directory.mkdir(mode=0o700)
            child = guard.launch([str(frozen.PYTHON), "-B", str(Path(__file__).resolve()), "--state", str(state), "--worker", policy],
                                 directory / "worker.log", env=environment)
            guard.wait(child, 30)
            results.append(json.loads((directory / "result.json").read_text()))
        result.update(compare(*results))
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        guard.close()
        result.update(wall_seconds=time.monotonic() - guard.started,
                      peak_sampled_footprint_bytes=max((s["footprint_bytes"] for s in guard.samples), default=0))
        try:
            frozen.verify(protocol["hashes"])
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
    mode.add_argument("--worker", choices=("A", "B"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.prepare:
        protocol = prepare(args.state)
        print(json.dumps({"state": str(args.state), "frozen_files": len(protocol["hashes"]), "inference_run": False}))
    elif args.run:
        execute(args.state)
    else:
        worker(args.state, args.worker)


if __name__ == "__main__":
    main()
