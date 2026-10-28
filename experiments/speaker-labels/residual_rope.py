#!/usr/bin/env python3
"""Bound residual-prefix and native output-head comparisons."""

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
ROTARY_PROTOCOL = ROOT / "residual-operator-20261028/protocol.json"
ROTARY_PROTOCOL_SHA = "8f0a8df6a78d69cd50f2e9e730825724b5313e1e0183efe7d7b1fce4b27a62c5"
SILU_PROPOSAL = Path("/tmp/unified-coordinator-lalia-speaker-labels/next-silu-proposal-20261028.md")
SILU_PROVENANCE = SILU_PROPOSAL.with_name("next-silu-source-provenance-20261028.json")
CUDA_INPUTS = Path("/tmp/unified-coordinator-lalia-speaker-labels/exact-prefix-20261027/cuda-candidate-1/inputs")
CUDA_ARCHIVE_PINS = {
    "inputs/manifest.json": "ebf3fbd30d2326f31ad8ed0eed9ce17fe4c3fccdaa4f0774278a8309ff092286",
    "probe/score_decoder.py": "8c5eca8e39514354be86c13aed499833ed11d9901f640d264f0d573e85262e42",
    "execution/remote/results/score/model/result.json": "b012ef49980d27db321d392f730e6d1edfd425ef41b6221f8ec5344419e2059a",
}
BOUNDARY_CUDA = ("score_decoder.py", "supervise.py", "smoke_numeric.py", "hidden_kernel_preflight.py",
                 "official_kernel_preflight.py", "official_transcribe.py")


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


def silu_provenance():
    """Freeze verified source evidence without importing the model libraries."""
    from importlib.metadata import distribution

    provenance = json.loads(SILU_PROVENANCE.read_text())
    source = Path(provenance["source_directory"])
    installed = Path(provenance["installed_site_packages"])
    hashes = {str(source / entry["file"]): entry["sha256"] for entry in provenance["upstream"]}
    hashes.update({str(source / name): provenance["probe"][name] for name in ("check.py", "check.json")})
    hashes.update({str(installed / name): digest for name, digest in provenance["installed_source_hashes"].items()})
    hashes[str(ROTARY_PROTOCOL)] = ROTARY_PROTOCOL_SHA
    verify(hashes)
    for name, version in provenance["installed_versions_from_metadata"].items():
        package = distribution(name)
        if package.version != version or Path(package.locate_file("")).resolve() != installed.resolve():
            raise ValueError(f"SiLU {name} metadata differs from reviewed runtime")
        path = installed / f"{name}-{version}.dist-info/METADATA"
        if path.read_text() != package.read_text("METADATA"):
            raise ValueError(f"SiLU {name} metadata file differs")
        hashes[str(path)] = sha256(path)
    hashes.update({str(path): sha256(path) for path in (SILU_PROPOSAL, SILU_PROVENANCE)})
    return hashes


def archived_cuda_inputs():
    """Bind the pre-profile manifest to its executed scorer and recorded schedule."""
    from hidden_boundary import RAW_SHA as head_raw_sha

    root = CUDA_INPUTS.parent
    hashes = {str(root / name): digest for name, digest in CUDA_ARCHIVE_PINS.items()}
    hashes[str(HISTORICAL / "protocol.json")] = HISTORICAL_SHA
    verify(hashes)
    protocol = json.loads((HISTORICAL / "protocol.json").read_text())
    result = json.loads((root / "execution/remote/results/score/model/result.json").read_text())
    manifest = json.loads((CUDA_INPUTS / "manifest.json").read_text())
    scorer = CUDA_ARCHIVE_PINS["probe/score_decoder.py"]
    if (not any(name.endswith("/cuda/score_decoder.py") and digest == scorer
                for name, digest in protocol["hashes"].items())
            or (protocol["consumed_tokens"], protocol["chunk_tokens"], protocol["final_position"]) != (TOTAL, CHUNK, TOTAL - 1)
            or (result["consumed"], result["cache_tokens"], result["chunk"], result["final_position"]) != (TOTAL, TOTAL, CHUNK, TOTAL - 1)
            or result["probe_sha256"] != scorer or result["fp32_rope"] is not False
            or result["rope_dtype"] != "torch.bfloat16" or result["raw_bf16_sha256"] != head_raw_sha["cuda"]
            or result["logits_file_sha256"] != CUDA_SHA or result["model_revision"] != MODEL_REVISION
            or manifest["model_revision"] != MODEL_REVISION or manifest["embedding_float32_sha256"] != CONTENT_SHA):
        raise ValueError("Archived CUDA exact-prefix provenance differs")
    inputs = {str(CUDA_INPUTS / name): digest for name, digest in manifest["files"].items()}
    verify(inputs)
    hashes.update(inputs)
    return manifest, hashes


def prepare(state, activation=None, boundary=None, capture_state=None, cuda_state=None):
    """Hash existing evidence and copy reviewed sources. Never import model libraries."""
    if activation not in (None, "full-silu-bf16"):
        raise ValueError("Unknown activation comparison")
    if boundary not in (None, "capture", "heads") or (boundary and activation):
        raise ValueError("Choose one fixed comparison")
    if ((boundary == "heads" and (capture_state is None or cuda_state is None))
            or (boundary != "heads" and (capture_state is not None or cuda_state is not None))):
        raise ValueError("Head replay requires both completed capture directories")
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
    helpers = HELPERS + (("residual_silu.py", "test_residual_silu.py") if activation else ())
    if boundary:
        helpers += ("hidden_boundary.py", "test_hidden_boundary.py")
        helpers += tuple("cuda/" + name for name in BOUNDARY_CUDA)
    paths += [ARCHIVE, CUDA, PYTHON.resolve(), *(SOURCE / name for name in helpers + PROTECTED)]
    if boundary:
        paths.append(Path("/tmp/unified-coordinator-lalia-speaker-labels/next-hidden-boundary-proposal-20261028.md"))
        _, archive_hashes = archived_cuda_inputs()
        hashes.update(archive_hashes)
    if boundary == "heads":
        paths += boundary_inputs(capture_state, cuda_state)
    hashes.update({str(path): sha256(path) for path in paths})
    if activation:
        hashes.update(silu_provenance())
    verify({str(trace / "combined-embeddings.safetensors"): EMBEDDING_SHA})
    state.mkdir(parents=True, mode=0o700)
    source = state / "source"
    source.mkdir()
    for name in helpers:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, source / name)
        hashes[str(source / name)] = hashes[str(SOURCE / name)]
    scope = "Residual-prefix SiLU A/B" if activation else "Residual-prefix whole-RoPE A/B"
    if boundary:
        scope = "Final hidden boundary " + boundary
    arms = ("head",) if boundary == "heads" else (("baseline",) if boundary == "capture" else ("baseline", "candidate"))
    protocol = {"scope": scope + ". No token generation or CUDA execution.", "activation": activation,
                "boundary": boundary, "arms": list(arms),
                "capture_state": str(capture_state) if capture_state else None,
                "cuda_state": str(cuda_state) if cuda_state else None,
                "model_revision": MODEL_REVISION, "trace": str(trace), "cuda": str(CUDA),
                "python_resolved": str(PYTHON.resolve()), "runtime_sources": runtime_sources,
                "relocations": relocations, "hashes": hashes,
                "schedule": {"prompt": 71687, "generated": 29448, "consumed": TOTAL,
                             "chunk": CHUNK, "final_position": TOTAL - 1},
                "operators": {"baseline": "NativeBF16RoPE(131072)",
                              "candidate": "NativeBF16RoPE(131072)" if activation else "ReferenceRoPE(131072, bf16_frequencies=True)"},
                "limits": {"arm_seconds": 300, "overall_seconds": 600, "term_grace_seconds": 5,
                           "work_seconds": WORK_SECONDS, "cleanup_seconds": 15,
                           "postflight_hash_seconds": 20, "postflight_health_seconds": 5,
                           "sampled_footprint_bytes": LIMIT, "battery_cutoff_percent": 25},
                "commands": {arm: [str(PYTHON), "-B", str(source / "residual_rope.py"),
                                    "--state", str(state), "--worker", arm]
                             for arm in (("silu-preflight", *arms) if activation else arms)}}
    if activation:
        protocol["limits"]["preflight_seconds"] = 30
    if boundary == "heads":
        protocol["limits"]["head_seconds"] = 60
    save(state / "protocol.json", protocol)
    (state / "protocol.sha256").write_text(sha256(state / "protocol.json") + "\n")
    return protocol


def completed_capture(state):
    from hidden_boundary import capture_receipt

    protocol = frozen(state)
    result = json.loads((state / "result.json").read_text())
    if (protocol.get("boundary") != "capture" or result["status"] != "complete"
            or result["cleanup"] is not True or result["frozen_files"] is not True
            or result["postflight_errors"] or result["production_after"] != 200):
        raise ValueError("Require completed native capture and cleanup")
    require_baseline(result["arms"]["baseline"])
    capture_receipt(state / "baseline", "mac")
    verify({str(state / "baseline/scores.safetensors"): result["arms"]["baseline"]["scores_sha256"]})
    return protocol


def boundary_inputs(capture_state, cuda_state):
    from hidden_boundary import capture_receipt

    completed_capture(capture_state)
    capture_receipt(cuda_state / "model", "cuda")
    result = json.loads((cuda_state / "model/result.json").read_text())
    lifecycle = json.loads((cuda_state / "lifecycle.json").read_text())
    required = {"owned_container_absent", "docker_attach_reaped", "service_start_completed", "service_health_ok",
                "hidden_boundary_verified", "service_idle_verified", "production_image_verified"}
    if (lifecycle["original_error"] or lifecycle["cleanup_errors"] or not required.issubset(lifecycle["events"])
            or result["consumed"] != TOTAL or result["chunk"] != CHUNK or result["cache_tokens"] != TOTAL
            or result["final_position"] != TOTAL - 1 or result["boundary"]["identical_hidden"]):
        raise ValueError("CUDA boundary or cleanup did not complete, or hidden states are identical")
    verify({str(cuda_state / "model/head-projections.safetensors"): result["boundary"]["projections_sha256"]})
    paths = [capture_state / name for name in ("protocol.json", "protocol.sha256", "result.json", "baseline-gate.json",
                                             "baseline/mac-head.json", "baseline/mac-head.safetensors", "baseline/scores.safetensors")]
    paths += [cuda_state / name for name in ("lifecycle.json", "model/result.json", "model/cuda-head.json",
                                           "model/cuda-head.safetensors", "model/head-projections.safetensors")]
    return paths


def export_hidden_cuda(state, output):
    completed_capture(state)
    if output.exists():
        raise ValueError("CUDA boundary bundle exists. No overwrite")
    manifest, _ = archived_cuda_inputs()
    inputs, probe = output / "inputs", output / "probe"
    inputs.mkdir(parents=True, mode=0o700)
    probe.mkdir(mode=0o700)
    for name in manifest["files"]:
        (inputs / name).symlink_to((CUDA_INPUTS / name).resolve())
    for name in ("mac-head.json", "mac-head.safetensors"):
        shutil.copy2(state / "baseline" / name, inputs / name)
        manifest["files"][name] = sha256(inputs / name)
    manifest["hidden_boundary"] = True
    manifest["score_profile"] = "exact-prefix"
    save(inputs / "manifest.json", manifest)
    (output / "input-manifest.sha256").write_text(sha256(inputs / "manifest.json") + "\n")
    for name in BOUNDARY_CUDA:
        shutil.copy2(SOURCE / "cuda" / name, probe / name)
    shutil.copy2(SOURCE / "hidden_boundary.py", probe / "hidden_boundary.py")
    save(output / "source-hashes.json", {path.name: sha256(path) for path in probe.iterdir()})


def select_operator(arm, activation=None):
    if activation not in (None, "full-silu-bf16") or arm not in ("baseline", "candidate"):
        raise ValueError("Unknown operator comparison")
    if activation:
        arm = "baseline"
    if arm == "baseline":
        rope = NativeBF16RoPE(131072)
        if rope.frequencies["denominator_sha256"] != DENOMINATOR_SHA:
            raise ValueError("Native denominators differ")
    else:
        rope = ReferenceRoPE(131072, bf16_frequencies=True)
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


def preflight_evidence(state):
    gate = json.loads((state / "silu-preflight-gate.json").read_text())
    directory = state / "silu-preflight"
    verify({str(directory / "result.json"): gate["result_sha256"],
            str(directory / "vectors.npz"): gate["vectors_sha256"]})
    report = json.loads((directory / "result.json").read_text())
    if report["status"] != "complete" or report["vectors_sha256"] != gate["vectors_sha256"]:
        raise ValueError("SiLU preflight did not complete")
    return report


def worker(state, arm):
    began = time.monotonic()
    protocol = frozen(state)
    if json.loads((state / "started.json").read_text())["pid"] != os.getppid():
        raise ValueError("Workers require the owning supervisor")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Require nice 10 or lower scheduling priority")
    activation = protocol.get("activation")
    if protocol.get("boundary") == "heads" and arm != "head":
        raise ValueError("Head comparison runs only the head worker")
    if arm == "head":
        if protocol.get("boundary") != "heads":
            raise ValueError("Head worker requires the fixed boundary comparison")
        import torch
        import mlx.core as mx
        from hidden_boundary import mlx_heads

        if mx.default_device() != mx.gpu:
            raise ValueError("Boundary head worker requires Metal")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        result = mlx_heads(MODEL / "model-00000-of-00001.safetensors",
                           Path(protocol["capture_state"]) / "baseline", Path(protocol["cuda_state"]) / "model", state / arm)
        save(state / arm / "result.json", result)
        return
    if protocol.get("boundary") == "capture" and arm != "baseline":
        raise ValueError("Boundary capture runs only the native baseline")
    if arm == "silu-preflight":
        if activation != "full-silu-bf16":
            raise ValueError("Preflight requires the SiLU comparison")
        from residual_silu import operator_preflight

        operator_preflight(state / arm, BASE / "runtime")
        return
    if activation:
        from residual_silu import require_preflight, select_activation

        require_preflight(preflight_evidence(state))
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

    if protocol.get("boundary") and mx.default_device() != mx.gpu:
        raise ValueError("Boundary capture requires Metal")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = load_model(MODEL, strict=True)
    module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if not module.is_relative_to(BASE / "runtime"):
        raise ValueError("Model imported outside historical runtime")
    config = model.config.text_config
    if (config.head_dim, config.rope_theta, config.max_position_embeddings, len(model.layers)) != (128, 1000000, 131072, 28):
        raise ValueError("Unexpected MOSS architecture")
    activation_report = select_activation(model, arm, BASE / "runtime") if activation else None
    if protocol.get("boundary") == "capture" and model.lm_head is not None:
        raise ValueError("Boundary capture requires the tied embedding head")
    rope = select_operator(arm, activation)
    for layer in model.layers:
        layer.self_attn.rope = rope
    tables = {}
    if arm == "candidate" and not activation:
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
                      activation=activation_report,
                      embedding_float32_sha256=digest.hexdigest(), comparisons=comparisons,
                      cuda_raw=summary(cuda), cuda_mlx_normalized=summary(cuda_normalized),
                      scores_sha256=sha256(directory / "scores.safetensors"),
                      elapsed_seconds=time.monotonic() - began)
        if protocol.get("boundary") == "capture":
            from hidden_boundary import capture_mlx

            require_baseline(result)
            result["head_capture"] = capture_mlx(final, raw, model.model.language_model.embed_tokens, directory)
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
    seconds = 30 if arm == "silu-preflight" else (60 if arm == "head" else 300)
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
        if time.monotonic() - began >= seconds:
            raise TimeoutError(f"{arm} exceeded {seconds}s including setup")
        time.sleep(0.5)
    child.wait(timeout=5)
    guard.check()
    if group_members(child.pid):
        raise RuntimeError("Worker left descendants. Stop before next arm.")
    guard.finished.add(child.pid)
    elapsed = time.monotonic() - began
    if child.returncode or elapsed >= seconds:
        raise RuntimeError(f"{arm} failed: exit {child.returncode}, elapsed {elapsed:.3f}s")
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError(f"{arm} did not complete")
    artifact, digest = ("vectors.npz", "vectors_sha256") if arm == "silu-preflight" else ("scores.safetensors", "scores_sha256")
    verify({str(directory / artifact): result[digest]})
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
        if protocol.get("activation"):
            measured = run_arm(state, "silu-preflight", protocol, guard)
            result["preflight"] = measured
            frozen(state)
            save(state / "silu-preflight-gate.json", {
                "result_sha256": sha256(state / "silu-preflight/result.json"),
                "vectors_sha256": measured["vectors_sha256"]})
        for arm in protocol.get("arms", ("baseline", "candidate")):
            guard.check()
            if protocol.get("activation"):
                preflight_evidence(state)
            measured = run_arm(state, arm, protocol, guard)
            result["arms"][arm] = measured
            frozen(state)
            for finished, evidence in result["arms"].items():
                verify({str(state / finished / "scores.safetensors"): evidence["scores_sha256"]})
            if arm == "baseline":
                require_baseline(measured)
                save(state / "baseline-gate.json", {"passed": True, "scores_sha256": measured["scores_sha256"]})
        if protocol.get("activation"):
            preflight_evidence(state)
        if protocol.get("boundary") == "capture":
            from hidden_boundary import capture_receipt

            capture_receipt(state / "baseline", "mac")
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
    mode.add_argument("--export-hidden-cuda", type=Path)
    mode.add_argument("--worker", choices=("baseline", "candidate", "silu-preflight", "head"), help=argparse.SUPPRESS)
    parser.add_argument("--activation", choices=("full-silu-bf16",), help="Prepare the fixed SiLU comparison instead of rotary")
    parser.add_argument("--boundary", choices=("capture", "heads"))
    parser.add_argument("--capture-state", type=Path)
    parser.add_argument("--cuda-state", type=Path)
    args = parser.parse_args()
    if (args.activation or args.boundary or args.capture_state or args.cuda_state) and not args.prepare:
        parser.error("Comparison options require --prepare; later modes use the frozen protocol")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.prepare:
        protocol = prepare(args.state.resolve(), args.activation, args.boundary,
                           args.capture_state.resolve() if args.capture_state else None,
                           args.cuda_state.resolve() if args.cuda_state else None)
        print(json.dumps({"pins": len(protocol["hashes"]), "inference_run": False}))
    elif args.run:
        execute(args.state.resolve())
    elif args.export_hidden_cuda:
        export_hidden_cuda(args.state.resolve(), args.export_hidden_cuda.resolve())
    else:
        worker(args.state.resolve(), args.worker)


if __name__ == "__main__":
    main()
