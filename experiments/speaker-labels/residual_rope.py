#!/usr/bin/env python3
"""Two gated residual-prefix replays, changing only the whole rotary operator."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from eos_rope import (BASE, MODEL, MODEL_REVISION, PYTHON, BF16_FREQUENCY_SHA256,
                      NativeBF16RoPE, ReferenceRoPE, chunks, verify)
from exact_prefix import (CHUNK, DENOMINATOR_SHA, EOS, TOTAL, WIDTH, compare_vectors,
                          validate_trace, vector_summary)
from short_windows import Guard, power
from window_matching import footprint, interrupt, save, sha256

SOURCE = Path(__file__).resolve().parent
ROOT = BASE.parent
HISTORICAL = ROOT / "exact-prefix-20261027/candidate-1"
HISTORICAL_SHA = "a555251a3933e66dcc659034cabdbad1f8015f9a58883f0dfcc41812bde7ff62"
ARCHIVE = BASE / "native-bf16-api-20261027/source-archive/manifest.json"
ARCHIVE_SHA = "15000402de7467a4e18391c86fbb9f3e5b5bd04da03bb0546f7e068037b5cc07"
CUDA = Path("/tmp/unified-coordinator-lalia-speaker-labels/exact-prefix-20261027/cuda-candidate-1/execution/remote/results/score/model/logits.safetensors")
CUDA_SHA = "62f631a1dc9d4892dbab9206dccf3b1d9ea13be8bc714d100e1c60097892efaa"
EMBEDDING_SHA = "f731281dba162553879315cca8a1f3f9a5287564c38fecee020e6438b0c8ba37"
CONTENT_SHA = "40fb613b275886f3abc2d57f959cdf161f30aad53fb53c467afb83ee5f19e9d8"
RAW_SHA = "ac43a5b368d5b3e61eed4a8f89d7373571b283caeab9eebc6003f64a1713d8fc"
NORMALIZED_SHA = "87b3ebd05bc418633a521f58ae279e30b7bf4c3c5664b97bff5d6924977b9356"
LIMIT = 32 * 1024**3
WORK_SECONDS = 555  # Reserve 45s of the 600s total for bounded cleanup and verification.
HELPERS = ("residual_rope.py", "test_residual_rope.py", "exact_prefix.py", "eos_rope.py",
           "trace_runtime.py", "short_windows.py", "window_matching.py")
PROTECTED = ("run.py", "score.py", "qualification.py", "cuda/manifest.json")


def inventory(root):
    return sorted(str(path) for path in root.rglob("*.py"))


def read_protocol(state):
    verify({str(state / "protocol.json"): (state / "protocol.sha256").read_text().strip()})
    return json.loads((state / "protocol.json").read_text())


def frozen(state):
    protocol = read_protocol(state)
    verify(protocol["hashes"])
    if inventory(BASE / "runtime") != protocol["runtime_sources"]:
        raise ValueError("Historical runtime source inventory changed")
    if Path(sys.executable).resolve() != Path(protocol["python_resolved"]):
        raise ValueError("Interpreter differs from frozen runtime")
    return protocol


def prepare(state):
    """Hash existing evidence and copy reviewed sources. Never import model libraries."""
    if state.exists():
        raise ValueError("State already exists. No overwrite or automatic retry.")
    verify({str(HISTORICAL / "protocol.json"): HISTORICAL_SHA, str(ARCHIVE): ARCHIVE_SHA,
            str(CUDA): CUDA_SHA})
    previous = json.loads((HISTORICAL / "protocol.json").read_text())
    hashes = dict(previous["hashes"])
    # Resolve only manifest-backed historical relocations, preserving expected bytes.
    relocations = []
    for entry in json.loads(ARCHIVE.read_text()):
        verify({entry["archive"]: entry["sha256"]})
        hashes[entry["archive"]] = entry["sha256"]
        for original, expected in list(hashes.items()):
            if Path(original).resolve() == Path(entry["source"]).resolve() and not Path(original).exists():
                if expected != entry["sha256"]:
                    raise ValueError("Archive differs from historical source pin")
                del hashes[original]
                relocations.append({"source": original, "archive": entry["archive"], "sha256": expected})
    verify(hashes)
    runtime_sources = inventory(BASE / "runtime")
    if runtime_sources != sorted(name for name in hashes if name.startswith(str(BASE / "runtime") + "/") and name.endswith(".py")):
        raise ValueError("Historical runtime inventory differs")
    trace = Path(previous["trace"])
    status = json.loads((trace / "status.json").read_text())
    validate_trace(status, json.loads((trace / "token-ids.json").read_text()),
                   json.loads((trace / "eos.json").read_text()))
    if (status["combined_embeddings"]["sha256"], status["combined_embeddings"]["float32_content_sha256"]) != (EMBEDDING_SHA, CONTENT_SHA):
        raise ValueError("Unexpected residual embeddings")
    if json.loads((BASE / "preflight.json").read_text())["model_revision"][1] != MODEL_REVISION:
        raise ValueError("Model revision differs")
    paths = [HISTORICAL / name for name in ("protocol.json", "protocol.sha256", "result.json", "scores.safetensors", "cuda-normalized-comparison.json")]
    paths += [ARCHIVE, CUDA, PYTHON.resolve(), *(SOURCE / name for name in HELPERS + PROTECTED)]
    hashes.update({str(path): sha256(path) for path in paths})
    verify({str(trace / "combined-embeddings.safetensors"): EMBEDDING_SHA})
    state.mkdir(parents=True, mode=0o700)
    source = state / "source"
    source.mkdir()
    for name in HELPERS:
        shutil.copy2(SOURCE / name, source / name)
        hashes[str(source / name)] = hashes[str(SOURCE / name)]
    protocol = {"scope": "Residual-prefix whole-RoPE A/B. No token generation or CUDA execution.",
                "model_revision": MODEL_REVISION, "trace": str(trace), "cuda": str(CUDA),
                "python_resolved": str(PYTHON.resolve()), "runtime_sources": runtime_sources,
                "relocations": relocations, "hashes": hashes,
                "schedule": {"prompt": 71687, "generated": 29448, "consumed": TOTAL,
                             "chunk": CHUNK, "final_position": TOTAL - 1},
                "operators": {"baseline": "NativeBF16RoPE(131072)",
                              "candidate": "ReferenceRoPE(131072, bf16_frequencies=True)"},
                "limits": {"arm_seconds": 300, "overall_seconds": 600, "term_grace_seconds": 5,
                           "work_seconds": WORK_SECONDS, "cleanup_seconds": 15,
                           "postflight_hash_seconds": 20, "postflight_health_seconds": 5,
                           "sampled_footprint_bytes": LIMIT, "battery_cutoff_percent": 25},
                "commands": {arm: [str(PYTHON), "-B", str(source / "residual_rope.py"),
                                    "--state", str(state), "--worker", arm]
                             for arm in ("baseline", "candidate")}}
    save(state / "protocol.json", protocol)
    (state / "protocol.sha256").write_text(sha256(state / "protocol.json") + "\n")
    return protocol


def select_operator(arm):
    if arm == "baseline":
        rope = NativeBF16RoPE(131072)
        if rope.frequencies["denominator_sha256"] != DENOMINATOR_SHA:
            raise ValueError("Native denominators differ")
    elif arm == "candidate":
        rope = ReferenceRoPE(131072, bf16_frequencies=True)
    else:
        raise ValueError("Unknown arm")
    if rope.frequencies["stored_sha256"] != BF16_FREQUENCY_SHA256:
        raise ValueError("Stored BF16 inverse frequencies differ")
    return rope


def require_baseline(result):
    if (result["consumed_tokens"] != TOTAL or result["cache_offsets"] != [TOTAL] * 28
            or result["chunk_tokens"] != CHUNK or result["final_position"] != TOTAL - 1):
        raise ValueError("Baseline cache/schedule differs")
    for name, digest in (("raw_logits", RAW_SHA), ("replay", NORMALIZED_SHA)):
        summary = result[name]
        if (summary["raw_sha256"] != digest or summary["argmax_id"] != EOS
                or summary["dtype"] != "mlx.core.bfloat16" or summary["eos_margin"] != 0.25
                or summary["best_continuation_id"] != 3821 or summary["eos_minus_3821"] != 0.25):
            raise ValueError(f"Baseline {name} did not reproduce exact historical vector")


def summary(values):
    import mlx.core as mx

    if values.dtype != mx.bfloat16:
        raise ValueError("Require native BF16 score vectors")
    result = vector_summary(values)
    result.update(tie_count=int(mx.sum(values == mx.max(values)).item()),
                  token_3821_score=float(values[3821].item()),
                  eos_minus_3821=float(values[EOS].item()) - float(values[3821].item()))
    return result


def differences(left, right):
    import mlx.core as mx

    difference = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    return {"max_absolute_difference": float(mx.max(difference).item()),
            "mean_absolute_difference": float(mx.mean(difference).item())}


def worker(state, arm):
    began = time.monotonic()
    protocol = frozen(state)
    if json.loads((state / "started.json").read_text())["pid"] != os.getppid():
        raise ValueError("Workers require the owning supervisor")
    if arm == "candidate":
        baseline = json.loads((state / "baseline/result.json").read_text())
        require_baseline(baseline)
        gate = json.loads((state / "baseline-gate.json").read_text())
        if gate != {"passed": True, "scores_sha256": baseline["scores_sha256"]}:
            raise ValueError("Candidate requires the completed baseline gate")
        verify({str(state / "baseline/scores.safetensors"): gate["scores_sha256"]})
    directory = state / arm
    import mlx.core as mx
    import numpy as np
    import torch
    from mlx_audio.stt.utils import load_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Require nice 10 or lower scheduling priority")
    model = load_model(MODEL, strict=True)
    module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not module.is_relative_to(BASE / "runtime"):
        raise ValueError("Model imported outside historical runtime")
    config = model.config.text_config
    if (config.head_dim, config.rope_theta, config.max_position_embeddings, len(model.layers)) != (128, 1000000, 131072, 28):
        raise ValueError("Unexpected MOSS architecture")
    rope = select_operator(arm)
    for layer in model.layers:
        layer.self_attn.rope = rope
    tables = {}
    if arm == "candidate":
        for name in ("cos", "sin"):
            value = getattr(rope, name)
            tables[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                            "sha256": hashlib.sha256(np.array(value).tobytes()).hexdigest(),
                            "bf16_sha256": hashlib.sha256(np.array(value.astype(mx.bfloat16).view(mx.uint8)).tobytes()).hexdigest()}
    trace = Path(protocol["trace"])
    embeddings = mx.load(str(trace / "combined-embeddings.safetensors"))["inputs_embeds"]
    if embeddings.shape != (TOTAL, WIDTH) or embeddings.dtype != mx.bfloat16:
        raise ValueError("Unexpected captured embeddings")
    digest = hashlib.sha256()
    for start, end in chunks(TOTAL, CHUNK):
        digest.update(np.array(embeddings[start:end].astype(mx.float32)).tobytes())
    if digest.hexdigest() != CONTENT_SHA:
        raise ValueError("Embedding content differs")
    cache = model.make_cache()
    try:
        if len(cache) != 28:
            raise ValueError("Unexpected KV cache count")
        for start, end in chunks(TOTAL, CHUNK):
            if any(item.offset != start for item in cache):
                raise ValueError("KV offset differs from captured prefix")
            hidden = model.model.language_model(inputs_embeds=embeddings[None, start:end], cache=cache)
            mx.eval(hidden, [item.state for item in cache])
            mx.clear_cache()
            save(directory / "progress.json", {"consumed": end, "elapsed_seconds": time.monotonic() - began})
        offsets = [item.offset for item in cache]
        if offsets != [TOTAL] * 28:
            raise ValueError("Final cache offsets differ")
        cache_dtypes = [[str(value.dtype) for value in item.state] for item in cache]
        if any(dtype != "mlx.core.bfloat16" for pair in cache_dtypes for dtype in pair):
            raise ValueError("KV cache dtype differs")
        final = hidden[:, -1:, :]
        logits = model.model.language_model.embed_tokens.as_linear(final) if model.lm_head is None else model.lm_head(final)
        raw = logits[0, 0]
        mx.save_safetensors(str(directory / "scores.safetensors"), {"logits": raw})
        captured = mx.load(str(trace / "eos-logprobs.safetensors"))["logprobs"]
        result, normalized = compare_vectors(raw, captured)
        result.update(raw_logits=summary(raw), replay=summary(normalized))
        mx.save_safetensors(str(directory / "scores.safetensors"), {"logits": raw, "logprobs": normalized})
        cuda = mx.load(protocol["cuda"])["logits"]
        cuda_normalized = (cuda[None] - mx.logsumexp(cuda[None], keepdims=True)).squeeze(0)
        comparisons = {"archived_cuda": {"raw": differences(raw, cuda),
                                         "native_normalized": differences(normalized, cuda_normalized)}}
        if arm == "candidate":
            verify({str(state / "baseline/scores.safetensors"): gate["scores_sha256"]})
            baseline = mx.load(str(state / "baseline/scores.safetensors"))
            comparisons["baseline"] = {"raw": differences(raw, baseline["logits"]),
                                       "native_normalized": differences(normalized, baseline["logprobs"])}
        result.update(status="complete", consumed_tokens=TOTAL, cache_offsets=offsets, cache_dtypes=cache_dtypes, chunk_tokens=CHUNK,
                      final_position=TOTAL - 1, frequencies=rope.frequencies, tables=tables,
                      embedding_float32_sha256=digest.hexdigest(), comparisons=comparisons,
                      cuda_raw=summary(cuda), cuda_mlx_normalized=summary(cuda_normalized),
                      scores_sha256=sha256(directory / "scores.safetensors"),
                      elapsed_seconds=time.monotonic() - began)
        save(directory / "result.json", result)
    finally:
        cache.clear()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def group_members(group):
    output = subprocess.run(["ps", "-axo", "pid=,pgid="], check=True, capture_output=True,
                            text=True, timeout=5).stdout
    return [int(pid) for pid, pgid in (line.split() for line in output.splitlines()) if int(pgid) == group]


class ReplayGuard(Guard):
    """Retain existing power/deadline sampling and account for whole owned groups."""

    def __init__(self, state, footprint_limit=LIMIT):
        super().__init__(state, footprint_limit)
        self.interrupted = []
        self.finished = set()

    def check(self):
        if self.interrupted:
            raise KeyboardInterrupt(f"Interrupted by {signal.Signals(self.interrupted[0]).name}")
        super().check()
        extra = 0
        leaders = {child.pid for child in self.children}
        for child in self.children:
            if child.pid in self.finished:
                continue
            for pid in group_members(child.pid):
                if pid not in leaders:
                    try:
                        extra += footprint(pid)
                    except ProcessLookupError:
                        if pid in group_members(child.pid):
                            raise
        self.samples[-1]["footprint_bytes"] += extra
        if self.samples[-1]["footprint_bytes"] > self.footprint_limit:
            raise MemoryError("Combined task-child footprint exceeded 32GiB")

    def close(self):
        errors = []
        for child in reversed(self.children):
            if child.pid in self.finished:
                continue
            try:
                try:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    until = time.monotonic() + 5
                    while group_members(child.pid) and time.monotonic() < until:
                        child.poll()
                        time.sleep(0.05)
                finally:
                    # Even a failed process inspection must not leave the owned group running.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)
                if group_members(child.pid):
                    raise RuntimeError(f"Owned group {child.pid} remains after cleanup")
            except Exception as error:
                errors.append(f"{child.pid}: {type(error).__name__}: {error}")
        save(self.state / "memory.json", self.samples)
        save(self.state / "cleanup.json", {"children": [{"pid": p.pid, "returncode": p.returncode}
                                                       for p in self.children], "errors": errors})
        if errors:
            raise RuntimeError("; ".join(errors))


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"Production HTTP status {response.status}")
        return response.status


def deadline(*_):
    raise TimeoutError("Replay work/finalization deadline")


def run_arm(state, arm, protocol, guard):
    began = time.monotonic()
    guard.check()
    directory = state / arm
    directory.mkdir()
    env = {**os.environ, "PYTHONPATH": str(BASE / "runtime"), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
    child = guard.launch(protocol["commands"][arm], directory / "worker.log", env=env)
    sample_start = len(guard.samples)
    while child.poll() is None:
        guard.check()
        if time.monotonic() - began >= 300:
            raise TimeoutError(f"{arm} exceeded 300s including setup")
        time.sleep(0.5)
    child.wait(timeout=5)
    guard.check()
    if group_members(child.pid):
        raise RuntimeError("Worker left descendants. Stop before next arm.")
    guard.finished.add(child.pid)
    elapsed = time.monotonic() - began
    if child.returncode or elapsed >= 300:
        raise RuntimeError(f"{arm} failed: exit {child.returncode}, elapsed {elapsed:.3f}s")
    result = json.loads((directory / "result.json").read_text())
    verify({str(directory / "scores.safetensors"): result["scores_sha256"]})
    result.update(wall_seconds=elapsed, worker_pid=child.pid, reaped=True,
                  peak_sampled_footprint_bytes=max((s["footprint_bytes"] for s in guard.samples[sample_start:]), default=0))
    save(directory / "result.json", result)
    return result


def execute(state):
    guard = ReplayGuard(state, footprint_limit=LIMIT)
    result = {"status": "failed", "arms": {}}
    with (state / "started.json").open("x") as marker:
        json.dump({"pid": os.getpid(), "unix_seconds": time.time()}, marker)
    interrupted = guard.interrupted

    def cancel(number, _):
        interrupted.append(number)

    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, cancel)
    signal.signal(signal.SIGALRM, deadline)
    previous_alarm = signal.alarm(WORK_SECONDS)
    try:
        result["production_before"] = health()
        result["power_before"] = power()
        protocol = frozen(state)
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            raise ValueError("Launch through nice -n 10")
        for arm in ("baseline", "candidate"):
            guard.check()
            measured = run_arm(state, arm, protocol, guard)
            result["arms"][arm] = measured
            frozen(state)
            for finished, evidence in result["arms"].items():
                verify({str(state / finished / "scores.safetensors"): evidence["scores_sha256"]})
            if arm == "baseline":
                require_baseline(measured)
                save(state / "baseline-gate.json", {"passed": True, "scores_sha256": measured["scores_sha256"]})
        result["status"] = "complete"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        failures = []
        # Reserved time keeps timeout failures subject to the same postflight checks.
        signal.alarm(0)
        for name, seconds, action in (("cleanup", 15, guard.close),
                                      ("frozen_files", 20, lambda: frozen(state)),
                                      ("production_after", 5, health)):
            try:
                signal.alarm(seconds)
                value = action()
                result[name] = value if name == "production_after" else True
            except BaseException as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
                result[name] = False
            finally:
                signal.alarm(0)
        elapsed = time.monotonic() - guard.started
        if elapsed >= 600:
            failures.append("Overall 600s bound exceeded, including postflight")
        if interrupted:
            failures.append("Interrupted: " + ", ".join(signal.Signals(number).name for number in interrupted))
        result.update(wall_seconds=elapsed, postflight_errors=failures)
        if failures:
            result["status"] = "failed"
        save(state / "result.json", result)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        if previous_alarm:
            signal.alarm(max(1, previous_alarm - int(elapsed)))
        if failures:
            raise RuntimeError("; ".join(failures))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.prepare:
        protocol = prepare(args.state.resolve())
        print(json.dumps({"pins": len(protocol["hashes"]), "inference_run": False}))
    elif args.run:
        execute(args.state.resolve())
    else:
        worker(args.state.resolve(), args.worker)


if __name__ == "__main__":
    main()
