"""One fixed decoder cut, preserving every prefix position and native KV cache."""

import json
from pathlib import Path

from hidden_boundary import (RAW_SHA, SHAPES, VOCAB, WEIGHT_SHA, WIDTH,
                             mlx_digest, require_tensor, save, sha256, torch_digest)

TOTAL, CHUNK, CUT = 101135, 4096, 14
PHASES = ("capture", "self", "cross")
HIDDEN_SHA = {"mac": "10b2a7d1726b7dd607b5364f5e0746df41170096d2543d30435698a06605b168",
              "cuda": "8e05857b38e22ea26ab4415e497c6fde6e69b3d0e9924e2259cac507e9c2c5a4"}
NORMALIZED_SHA = "87b3ebd05bc418633a521f58ae279e30b7bf4c3c5664b97bff5d6924977b9356"
PARENT = Path("/tmp/unified-coordinator-lalia-speaker-labels/hidden-boundary-20261028")
PARENT_REPORT_SHA = "d115266840c457d6227b4623823fe148df81146e4bab00857ef306a2b3ac6044"
HEAD_RECEIPTS = {
    "mac": (Path.home() / ".local/state/lalia-speaker-qualification/hidden-boundary-20261028/capture/baseline/mac-head.json",
            "486848a8de881bc91ef14101754ba06ff3fa1258e153430badadba46b241e9f9"),
    "cuda": (PARENT / "remote/results/score/model/cuda-head.json",
             "b6444b9b592f05977de96bf6aa5920e66c1a4e8627a85c597b3b73d4b300a480"),
}
PROPOSAL = PARENT.parent / "next-decoder-boundary-proposal-20261028.md"
PROPOSAL_SHA = "c1c63cd2e9424204a5c8d68b2a20fbf6095dd00bb0e8f276faab03aa4822f321"


def intervals():
    return [(start, min(start + CHUNK, TOTAL - 1)) for start in range(0, TOTAL - 1, CHUNK)] + [(TOTAL - 1, TOTAL)]


def verify_files(hashes):
    for name, digest in hashes.items():
        if sha256(name) != digest:
            raise ValueError(f"Midpoint artifact changed: {name}")


def historical_pins():
    """Use immutable measured receipts, not prior protocols' mutable repo paths."""
    hashes = {str(PARENT / "execution-report.json"): PARENT_REPORT_SHA, str(PROPOSAL): PROPOSAL_SHA}
    verify_files(hashes)
    report = json.loads((PARENT / "execution-report.json").read_text())
    if (report["status"] != "complete" or not report["valid_comparison"]
            or not report["native_self_projections_exact"] or report["weight_bf16_sha256"] != WEIGHT_SHA):
        raise ValueError("Parent boundary comparison did not complete")
    for backend, name in (("mac", "mac_head_mac_hidden"), ("cuda", "cuda_head_cuda_hidden")):
        if report["table"][name]["raw_bf16_sha256"] != RAW_SHA[backend]:
            raise ValueError("Parent full-vector reference differs")
        path, expected = HEAD_RECEIPTS[backend]
        verify_files({str(path): expected})
        receipt = json.loads(path.read_text())
        if (receipt["backend"] != backend or receipt["hidden_sha256"] != HIDDEN_SHA[backend]
                or receipt["logits_sha256"] != RAW_SHA[backend] or receipt["weight_sha256"] != WEIGHT_SHA):
            raise ValueError("Parent final-hidden reference differs")
        hashes.update({str(path): expected, str(path.with_suffix(".safetensors")): receipt["file_sha256"]})
    verify_files(hashes)
    return hashes


def read_sequence(directory, backend):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["backend"] != backend or manifest["cut"] != CUT or manifest["dtype"] != "BF16"
            or manifest["width"] != WIDTH or [(r["start"], r["end"]) for r in manifest["chunks"]] != intervals()):
        raise ValueError("Midpoint sequence contract differs")
    expected = {"manifest.json"}
    for index, row in enumerate(manifest["chunks"]):
        name = f"{index:02d}.safetensors"
        if row["file"] != name or row["shape"] != [1, row["end"] - row["start"], WIDTH]:
            raise ValueError("Midpoint chunk name or shape differs")
        expected.add(name)
        if sha256(directory / name) != row["file_sha256"]:
            raise ValueError("Midpoint chunk file changed")
    if {p.name for p in directory.iterdir()} != expected or any(not p.is_file() or p.is_symlink() for p in directory.iterdir()):
        raise ValueError("Midpoint sequence inventory differs")
    return manifest


def final_receipt(directory, backend, phase):
    result = json.loads((directory / "final.json").read_text())
    if (result["backend"] != backend or result["phase"] != phase or result["cut"] != CUT
            or result["weight_sha256"] != WEIGHT_SHA or result["hidden_shape"] != list(SHAPES[backend])
            or result["dtype"] != "BF16" or sha256(directory / "final.safetensors") != result["file_sha256"]):
        raise ValueError("Midpoint final tensor receipt differs")
    if phase != "cross" and (result["raw_sha256"] != RAW_SHA[backend] or result["hidden_sha256"] != HIDDEN_SHA[backend]):
        raise ValueError("Midpoint native full-vector/final-hidden gate failed")
    return result


def completed_run(state, backend, phase):
    """Verify the required result and cleanup before freezing a dependent pass."""
    if backend == "mac":
        result = json.loads((state / "result.json").read_text())
        if (result["status"] != "complete" or result["cleanup"] is not True or result["frozen_files"] is not True
                or result["postflight_errors"] or result["production_after"] != 200):
            raise ValueError("Mac midpoint result or cleanup failed")
        directory = state / "midpoint"
        report = json.loads((directory / "result.json").read_text())
        if (report["status"] != "complete" or report["consumed_tokens"] != TOTAL
                or report["cache_offsets"] != [TOTAL] * 28 or report["chunk_tokens"] != CHUNK
                or report["final_position"] != TOTAL - 1):
            raise ValueError("Mac midpoint cache/schedule differs")
        if phase != "cross" and report["replay"]["raw_sha256"] != NORMALIZED_SHA:
            raise ValueError("Mac midpoint normalized vector differs")
        artifact, digest = "scores.safetensors", report["scores_sha256"]
        files = [state / "result.json"]
    else:
        lifecycle = json.loads((state / "lifecycle.json").read_text())
        required = {"owned_container_absent", "docker_attach_reaped", "service_start_completed", "service_health_ok",
                    "midpoint_verified", "service_idle_verified", "production_image_verified"}
        if lifecycle["original_error"] or lifecycle["cleanup_errors"] or not required.issubset(lifecycle["events"]):
            raise ValueError("CUDA midpoint cleanup failed")
        directory = state / "model"
        report = json.loads((directory / "result.json").read_text())
        if (report["consumed"] != TOTAL or report["cache_tokens"] != TOTAL or report["chunk"] != CHUNK
                or report["final_position"] != TOTAL - 1):
            raise ValueError("CUDA midpoint cache/schedule differs")
        artifact, digest = "logits.safetensors", report["logits_file_sha256"]
        files = [state / "lifecycle.json"]
    measured = report["midpoint"]
    if measured["phase"] != phase or measured["backend"] != backend or measured["chunks"] != len(intervals()):
        raise ValueError("Midpoint phase result differs")
    final = final_receipt(directory, backend, phase)
    if measured["final"] != final or sha256(directory / artifact) != digest:
        raise ValueError("Midpoint result tensors changed")
    files += [directory / name for name in ("result.json", "final.json", "final.safetensors", artifact)]
    if phase == "capture":
        read_sequence(directory / "sequence", backend)
        if sha256(directory / "sequence/manifest.json") != measured["sequence_sha256"]:
            raise ValueError("Captured midpoint manifest changed")
        files += sorted((directory / "sequence").iterdir())
    return directory, measured, files


def phase_config(root, backend, phase, source=None, mac_self=None, cuda_self=None):
    if backend not in ("mac", "cuda") or phase not in PHASES:
        raise ValueError("Unknown midpoint pass")
    if (phase == "capture") != (source is None):
        raise ValueError("Only capture omits the midpoint source")
    if backend == "mac" and phase != "cross" and (mac_self or cuda_self):
        raise ValueError("Early Mac passes have no self-replay gates")
    if backend == "cuda" and mac_self is None or phase == "cross" and (mac_self is None or cuda_self is None):
        raise ValueError("Missing prerequisite self-replay gate")
    if phase != "cross" and cuda_self is not None:
        raise ValueError("CUDA self gate belongs only to cross passes")
    files, source_manifest = [], None
    donor = backend if phase == "self" else ("cuda" if backend == "mac" else "mac")
    if source is not None:
        directory, measured, paths = completed_run(source, donor, "capture")
        files += paths
        source_manifest = measured["sequence_sha256"]
        sequence = directory / "sequence"
    else:
        sequence = None
    gates = {}
    for name, state in (("mac", mac_self), ("cuda", cuda_self)):
        if state is not None:
            _, measured, paths = completed_run(state, name, "self")
            files += paths
            gates[name] = measured
    if phase == "cross" and gates[donor]["input_manifest_sha256"] != source_manifest:
        raise ValueError("Cross input differs from the donor's self-replayed capture")
    hashes = {str(p.resolve().relative_to(root.resolve())): sha256(p) for p in files}
    return {"backend": backend, "phase": phase, "cut": CUT,
            "source": str(source.resolve().relative_to(root.resolve())) if source else None,
            "mac_self": str(mac_self.resolve().relative_to(root.resolve())) if mac_self else None,
            "cuda_self": str(cuda_self.resolve().relative_to(root.resolve())) if cuda_self else None,
            "source_backend": donor if source is not None else None,
            "sequence": str(sequence.resolve().relative_to(root.resolve())) if sequence else None,
            "input_manifest_sha256": source_manifest, "hashes": hashes}


def validate_config(root, config, backend=None):
    if (config["backend"] not in ("mac", "cuda") or config["phase"] not in PHASES or config["cut"] != CUT
            or backend is not None and config["backend"] != backend):
        raise ValueError("Midpoint pass configuration differs")
    hashes = {}
    for name, digest in config["hashes"].items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Midpoint prerequisite escapes the trial")
        hashes[str(path)] = digest
    verify_files(hashes)
    paths = {}
    for name in ("source", "mac_self", "cuda_self"):
        value = config[name]
        path = root / value if value is not None else None
        if path is not None and (Path(value).is_absolute() or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError("Midpoint prerequisite escapes the trial")
        paths[name] = path
    if phase_config(root, config["backend"], config["phase"], **paths) != config:
        raise ValueError("Midpoint prerequisite gates changed")
    if config["phase"] == "capture":
        if config["sequence"] is not None or config["input_manifest_sha256"] is not None:
            raise ValueError("Capture cannot substitute a sequence")
    else:
        sequence = root / config["sequence"]
        if not sequence.resolve().is_relative_to(root.resolve()) or sha256(sequence / "manifest.json") != config["input_manifest_sha256"]:
            raise ValueError("Midpoint input manifest changed")
        read_sequence(sequence, config["source_backend"])
    return hashes


class Midpoint:
    def __init__(self, root, config, directory):
        validate_config(root, config)
        self.backend, self.phase = config["backend"], config["phase"]
        self.root, self.config, self.directory = root, config, directory
        self.source = root / config["sequence"] if config["sequence"] else None
        self.manifest = read_sequence(self.source, config["source_backend"]) if self.source else None
        self.rows, self.index, self.pending, self.called = [], 0, None, False
        if self.phase == "capture":
            (directory / "sequence").mkdir(mode=0o700)

    def begin(self, start, end):
        if self.called or self.index >= len(intervals()) or (start, end) != intervals()[self.index]:
            raise ValueError("Midpoint chunk order differs")
        self.start, self.end = start, end

    def intercept(self, hidden):
        if self.called or (self.start, self.end) != intervals()[self.index]:
            raise ValueError("Midpoint block called more than once")
        self.called = True
        shape = (1, self.end - self.start, WIDTH)
        if tuple(hidden.shape) != shape:
            raise ValueError("Midpoint input shape differs")
        if self.phase == "capture":
            self.pending = hidden
            return hidden
        row = self.manifest["chunks"][self.index]
        path = self.source / row["file"]
        if sha256(path) != row["file_sha256"]:
            raise ValueError("Midpoint input changed during replay")
        if self.backend == "mac":
            import mlx.core as mx

            if hidden.dtype != mx.bfloat16:
                raise ValueError("Midpoint native dtype differs")
            value = mx.load(str(path))["hidden"]
            require_tensor(mlx_digest, value, shape, row["raw_sha256"])
        else:
            import torch
            from safetensors.torch import load_file

            if hidden.dtype != torch.bfloat16:
                raise ValueError("Midpoint native dtype differs")
            value = load_file(path, device=str(hidden.device))["hidden"]
            require_tensor(torch_digest, value, shape, row["raw_sha256"])
            if value.stride() != hidden.stride():
                raise ValueError("Midpoint CUDA replay stride differs")
        return value

    def finish(self):
        if not self.called:
            raise ValueError("Midpoint block was not called")
        if self.phase == "capture":
            path = self.directory / "sequence" / f"{self.index:02d}.safetensors"
            value = self.pending
            shape = (1, self.end - self.start, WIDTH)
            if self.backend == "mac":
                import mlx.core as mx

                digest = require_tensor(mlx_digest, value, shape)
                mx.save_safetensors(str(path), {"hidden": value})
                stride = None
            else:
                from safetensors.torch import save_file

                digest = require_tensor(torch_digest, value, shape)
                stride = list(value.stride())
                save_file({"hidden": value.detach().cpu().contiguous()}, path)
            self.rows.append({"start": self.start, "end": self.end, "shape": list(shape), "stride": stride,
                              "file": path.name, "raw_sha256": digest, "file_sha256": sha256(path)})
        self.pending, self.called = None, False
        self.index += 1

    def complete(self, hidden, logits, weight):
        if self.index != len(intervals()) or self.called or self.pending is not None:
            raise ValueError("Midpoint sequence did not finish")
        digest = mlx_digest if self.backend == "mac" else torch_digest
        expected = self.phase != "cross"
        final = {"backend": self.backend, "phase": self.phase, "cut": CUT, "dtype": "BF16",
                 "hidden_shape": list(hidden.shape),
                 "hidden_sha256": require_tensor(digest, hidden, SHAPES[self.backend], HIDDEN_SHA[self.backend] if expected else None),
                 "raw_sha256": require_tensor(digest, logits, (VOCAB,), RAW_SHA[self.backend] if expected else None),
                 "weight_sha256": require_tensor(digest, weight, (VOCAB, WIDTH), WEIGHT_SHA)}
        path = self.directory / "final.safetensors"
        if self.backend == "mac":
            import mlx.core as mx

            mx.save_safetensors(str(path), {"hidden": hidden, "logits": logits})
        else:
            from safetensors.torch import save_file

            save_file({"hidden": hidden.detach().cpu().contiguous(), "logits": logits.detach().cpu().contiguous()}, path)
        final["file_sha256"] = sha256(path)
        save(self.directory / "final.json", final)
        result = {"backend": self.backend, "phase": self.phase, "chunks": self.index, "final": final,
                  "input_manifest_sha256": self.config["input_manifest_sha256"]}
        if self.phase == "capture":
            sequence = self.directory / "sequence"
            save(sequence / "manifest.json", {"backend": self.backend, "cut": CUT, "dtype": "BF16",
                                               "width": WIDTH, "chunks": self.rows})
            read_sequence(sequence, self.backend)
            result["sequence_sha256"] = sha256(sequence / "manifest.json")
        validate_config(self.root, self.config, self.backend)
        return result


def install_mlx(layers, observer):
    import mlx.nn as nn

    original = layers[CUT]

    class Cut(nn.Module):
        def __call__(self, hidden, *args, **kwargs):
            return original(observer.intercept(hidden), *args, **kwargs)

    layers[CUT] = Cut()
    return original


def install_torch(layers, observer):
    def before(_module, args, kwargs):
        if not args or "hidden_states" in kwargs:
            raise ValueError("Unexpected native decoder block call")
        return (observer.intercept(args[0]), *args[1:]), kwargs

    return layers[CUT].register_forward_pre_hook(before, with_kwargs=True)


def prepare_cuda(root, phase):
    if phase not in PHASES:
        raise ValueError("Unknown CUDA midpoint pass")
    path = root / f"midpoint-{phase}.json"
    if path.exists() or path.with_suffix(".sha256").exists():
        raise ValueError("Midpoint configuration exists. No overwrite or retry")
    source = {"capture": None, "self": root / "results/midpoint-capture",
              "cross": root / "prerequisites/mac-capture"}[phase]
    config = phase_config(root, "cuda", phase, source, root / "prerequisites/mac-self",
                          root / "results/midpoint-self" if phase == "cross" else None)
    save(path, config)
    path.with_suffix(".sha256").write_text(sha256(path) + "\n")
    return config


def analyze(mac_self, cuda_self, mac_cross, cuda_cross):
    """Compare the four completed suffix outputs on CPU, without generating tokens."""
    import numpy as np
    from safetensors.torch import load_file

    states = (("mac_self", mac_self, "mac", "self"), ("cuda_self", cuda_self, "cuda", "self"),
              ("mac_cross", mac_cross, "mac", "cross"), ("cuda_cross", cuda_cross, "cuda", "cross"))
    vectors, hidden, reports, table = {}, {}, {}, {}
    for name, state, backend, phase in states:
        directory, report, _ = completed_run(state, backend, phase)
        values = load_file(directory / "final.safetensors", device="cpu")
        final = report["final"]
        require_tensor(torch_digest, values["hidden"], SHAPES[backend], final["hidden_sha256"])
        require_tensor(torch_digest, values["logits"], (VOCAB,), final["raw_sha256"])
        raw = values["logits"].float().numpy().astype(np.float64)
        continuation = raw.copy()
        continuation[[151643, 151645]] = -np.inf
        best = int(continuation.argmax())
        normalized = raw - (raw.max() + np.log(np.exp(raw - raw.max()).sum()))
        table[name] = {"raw_bf16_sha256": final["raw_sha256"], "hidden_bf16_sha256": final["hidden_sha256"],
                       "argmax": int(raw.argmax()), "tie_count": int(np.count_nonzero(raw == raw.max())),
                       "eos": float(raw[151645]), "token_3821": float(raw[3821]), "best_continuation_id": best,
                       "eos_margin": float(raw[151645] - raw[best]),
                       "analysis_fp64_logprob_eos": float(normalized[151645]),
                       "analysis_fp64_logprob_3821": float(normalized[3821])}
        reports[name], vectors[name] = report, raw
        hidden[name] = values["hidden"].float().numpy().reshape(-1).astype(np.float64)
    for receiving, donor in (("mac", "cuda"), ("cuda", "mac")):
        if reports[receiving + "_cross"]["input_manifest_sha256"] != reports[donor + "_self"]["input_manifest_sha256"]:
            raise ValueError("Cross sequence differs from the donor self-replay")

    def difference(left, right):
        values = left - right
        return {"different_elements": int(np.count_nonzero(values)), "max_absolute_difference": float(np.abs(values).max()),
                "mean_absolute_difference": float(np.abs(values).mean())}

    distances = {}
    for name in ("mac_cross", "cuda_cross"):
        distances[name] = {reference: {"logits": difference(vectors[name], vectors[reference]),
                                      "hidden": difference(hidden[name], hidden[reference])}
                           for reference in ("mac_self", "cuda_self")}
    return {"table": table, "differences": distances,
            "normalization": "Shared CPU FP64 analysis, not native generation probabilities",
            "limit": "One fixed cut. Cross interactions do not identify the earliest divergence or a defective operation."}


if __name__ == "__main__":
    import argparse
    import os

    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-cuda", choices=PHASES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = prepare_cuda(args.root.resolve(), args.prepare_cuda)
    print(json.dumps({"phase": config["phase"], "pins": len(config["hashes"]), "inference_run": False}))
