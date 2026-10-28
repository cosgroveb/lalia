import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import torch
from safetensors.torch import save_file

import midpoint as probe
import residual_rope as replay


class ScheduleTests(unittest.TestCase):
    def test_schedule_includes_every_position_and_final_singleton(self):
        rows = probe.intervals()
        self.assertEqual(len(rows), 26)
        self.assertEqual((rows[0], rows[-1]), ((0, 4096), (101134, 101135)))
        self.assertEqual(sum(end - start for start, end in rows), 101135)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(rows, rows[1:])))


class MidpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mx.set_default_device, mx.default_device())
        mx.set_default_device(mx.cpu)
        torch.set_num_threads(2)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.weight = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        self.hidden = {"mac": torch.arange(4, dtype=torch.bfloat16).reshape(1, 1, 4),
                       "cuda": torch.arange(4, dtype=torch.bfloat16).reshape(1, 4) + 1}
        self.logits = {name: torch.nn.functional.linear(value, self.weight).reshape(-1)
                       for name, value in self.hidden.items()}
        self.enterContext(patch.multiple(probe, TOTAL=5, CHUNK=2, CUT=1, WIDTH=4, VOCAB=6,
                                         WEIGHT_SHA=probe.torch_digest(self.weight),
                                         SHAPES={name: tuple(value.shape) for name, value in self.hidden.items()},
                                         HIDDEN_SHA={name: probe.torch_digest(value) for name, value in self.hidden.items()},
                                         RAW_SHA={name: probe.torch_digest(value) for name, value in self.logits.items()}))

    def mlx(self, value):
        return mx.array(value.view(torch.uint16).numpy()).view(mx.bfloat16)

    def test_parent_hidden_constants_are_bound_to_immutable_receipts_and_tensors(self):
        report = self.root / "execution-report.json"
        proposal = self.root / "proposal.md"
        proposal.write_text("approved")
        probe.save(report, {"status": "complete", "valid_comparison": True, "native_self_projections_exact": True,
                           "weight_bf16_sha256": probe.WEIGHT_SHA,
                           "table": {name + "_head_" + name + "_hidden": {"raw_bf16_sha256": probe.RAW_SHA[name]}
                                     for name in ("mac", "cuda")}})
        heads = {}
        for name in ("mac", "cuda"):
            tensor = self.root / (name + "-head.safetensors")
            tensor.write_bytes(name.encode())
            path = tensor.with_suffix(".json")
            probe.save(path, {"backend": name, "hidden_sha256": probe.HIDDEN_SHA[name], "logits_sha256": probe.RAW_SHA[name],
                              "weight_sha256": probe.WEIGHT_SHA, "file_sha256": probe.sha256(tensor)})
            heads[name] = (path, probe.sha256(path))
        with patch.multiple(probe, PARENT=self.root, PARENT_REPORT_SHA=probe.sha256(report), PROPOSAL=proposal,
                            PROPOSAL_SHA=probe.sha256(proposal), HEAD_RECEIPTS=heads):
            self.assertEqual(len(probe.historical_pins()), 6)
            with patch.dict(probe.HIDDEN_SHA, {"mac": "wrong"}), self.assertRaisesRegex(ValueError, "final-hidden"):
                probe.historical_pins()
            (self.root / "cuda-head.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                probe.historical_pins()

    def run_receipt(self, backend, phase, source=None, mac_self=None, cuda_self=None, name=None):
        state = self.root / (name or f"{backend}-{phase}")
        directory = state / ("midpoint" if backend == "mac" else "model")
        directory.mkdir(parents=True)
        config = probe.phase_config(self.root, backend, phase, source, mac_self, cuda_self)
        observer = probe.Midpoint(self.root, config, directory)
        for start, end in probe.intervals():
            observer.begin(start, end)
            value = torch.arange((end - start) * 4, dtype=torch.bfloat16).reshape(1, end - start, 4)
            if backend == "cuda":
                value += 1
            observer.intercept(self.mlx(value) if backend == "mac" else value)
            observer.finish()
        tensors = (self.hidden[backend], self.logits[backend], self.weight)
        measured = observer.complete(*(self.mlx(v) for v in tensors)) if backend == "mac" else observer.complete(*tensors)
        if backend == "mac":
            artifact = directory / "scores.safetensors"
            save_file({"logits": self.logits[backend]}, artifact)
            report = {"status": "complete", "consumed_tokens": 5, "cache_offsets": [5] * 28,
                      "chunk_tokens": 2, "final_position": 4, "scores_sha256": probe.sha256(artifact),
                      "replay": {"raw_sha256": probe.NORMALIZED_SHA}, "midpoint": measured}
            probe.save(state / "result.json", {"status": "complete", "cleanup": True, "frozen_files": True,
                                                "postflight_errors": [], "production_after": 200})
        else:
            artifact = directory / "logits.safetensors"
            save_file({"logits": self.logits[backend]}, artifact)
            report = {"consumed": 5, "cache_tokens": 5, "chunk": 2, "final_position": 4,
                      "logits_file_sha256": probe.sha256(artifact), "midpoint": measured}
            probe.save(state / "lifecycle.json", {"original_error": None, "cleanup_errors": [],
                      "events": ["owned_container_absent", "docker_attach_reaped", "service_start_completed",
                                 "service_health_ok", "midpoint_verified", "service_idle_verified", "production_image_verified"]})
        probe.save(directory / "result.json", report)
        return state, config

    def native_runs(self):
        mac_capture, _ = self.run_receipt("mac", "capture")
        mac_self, _ = self.run_receipt("mac", "self", mac_capture)
        cuda_capture, _ = self.run_receipt("cuda", "capture", mac_self=mac_self)
        cuda_self, _ = self.run_receipt("cuda", "self", cuda_capture, mac_self)
        return mac_capture, mac_self, cuda_capture, cuda_self

    def test_full_sequence_crosses_both_backends_only_after_self_gates(self):
        mac_capture, mac_self, cuda_capture, cuda_self = self.native_runs()
        for backend, source in (("mac", cuda_capture), ("cuda", mac_capture)):
            state, config = self.run_receipt(backend, "cross", source, mac_self, cuda_self)
            _, report, files = probe.completed_run(state, backend, "cross")
            self.assertEqual(report["chunks"], 3)
            self.assertEqual(report["input_manifest_sha256"], config["input_manifest_sha256"])
            self.assertTrue(files)

    def test_cross_rejects_missing_or_failed_self_gate(self):
        mac_capture, mac_self, cuda_capture, cuda_self = self.native_runs()
        with self.assertRaisesRegex(ValueError, "self-replay gate"):
            probe.phase_config(self.root, "mac", "cross", cuda_capture, mac_self)
        path = cuda_self / "lifecycle.json"
        result = json.loads(path.read_text())
        result["cleanup_errors"] = ["container remains"]
        probe.save(path, result)
        with self.assertRaisesRegex(ValueError, "cleanup failed"):
            probe.phase_config(self.root, "mac", "cross", cuda_capture, mac_self, cuda_self)

    def test_manifest_rejects_gap_duplicate_and_extra_files(self):
        state, _ = self.run_receipt("mac", "capture")
        directory = state / "midpoint/sequence"
        path = directory / "manifest.json"
        original = path.read_text()
        for field, value in (("start", 1), ("file", "01.safetensors"), ("shape", [1, 1, 4])):
            manifest = json.loads(original)
            manifest["chunks"][0][field] = value
            probe.save(path, manifest)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract|name or shape"):
                probe.read_sequence(directory, "mac")
        path.write_text(original)
        (directory / "extra").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            probe.read_sequence(directory, "mac")

    def test_replay_rejects_changed_input_and_configuration_before_forward(self):
        capture, _ = self.run_receipt("mac", "capture")
        config = probe.phase_config(self.root, "mac", "self", capture)
        config["phase"] = "cross"
        with self.assertRaisesRegex(ValueError, "self-replay gate"):
            probe.validate_config(self.root, config)
        config["phase"] = "self"
        path = capture / "midpoint/sequence/00.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            probe.validate_config(self.root, config)

    def test_source_must_match_the_sequence_used_by_donor_self_replay(self):
        mac_capture, mac_self, cuda_capture, cuda_self = self.native_runs()
        path = cuda_self / "model/result.json"
        report = json.loads(path.read_text())
        report["midpoint"]["input_manifest_sha256"] = "wrong"
        probe.save(path, report)
        with self.assertRaisesRegex(ValueError, "donor"):
            probe.phase_config(self.root, "mac", "cross", cuda_capture, mac_self, cuda_self)

    def test_mlx_wrapper_changes_only_the_selected_instance_input(self):
        class Block(nn.Module):
            def __call__(self, hidden, *, cache):
                cache.append(hidden)
                return hidden + 1

        config = probe.phase_config(self.root, "mac", "capture")
        directory = self.root / "observer"
        directory.mkdir()
        observer = probe.Midpoint(self.root, config, directory)
        layers = [Block(), Block(), Block()]
        original, other = layers[1], layers[0]
        returned = probe.install_mlx(layers, observer)
        self.assertIs(returned, original)
        self.assertIs(layers[0], other)
        value = mx.ones((1, 2, 4), dtype=mx.bfloat16)
        cache = []
        observer.begin(0, 2)
        result = layers[1](value, cache=cache)
        mx.eval(result)
        observer.finish()
        self.assertIs(cache[0], value)
        self.assertTrue(bool(mx.all(result == 2).item()))
        self.assertIs(type(original), Block)

    def test_torch_hook_preserves_arguments_and_detaches(self):
        class Block(torch.nn.Module):
            def forward(self, hidden, *, marker):
                self.marker = marker
                return hidden + 1

        class Observer:
            def intercept(self, hidden):
                return hidden + 2

        layers = torch.nn.ModuleList([Block(), Block()])
        original = layers[1]
        hook = probe.install_torch(layers, Observer())
        marker = object()
        value = torch.ones((1, 2, 4), dtype=torch.bfloat16)
        self.assertTrue(torch.equal(layers[1](value, marker=marker), value + 3))
        self.assertIs(original.marker, marker)
        self.assertIs(layers[1], original)
        hook.remove()
        self.assertTrue(torch.equal(layers[1](value, marker=marker), value + 1))

    def test_chunk_order_count_dtype_and_final_baseline_fail_closed(self):
        config = probe.phase_config(self.root, "mac", "capture")
        directory = self.root / "observer"
        directory.mkdir()
        observer = probe.Midpoint(self.root, config, directory)
        with self.assertRaisesRegex(ValueError, "order"):
            observer.begin(2, 4)
        with self.assertRaisesRegex(ValueError, "did not finish"):
            observer.complete(None, None, None)
        observer.begin(0, 2)
        observer.intercept(mx.ones((1, 2, 4)))
        with self.assertRaisesRegex(ValueError, "finite BF16"):
            observer.finish()

    def test_native_gate_rejects_final_hidden_even_when_logits_match(self):
        state, _ = self.run_receipt("mac", "capture")
        path = state / "midpoint/final.json"
        report = json.loads(path.read_text())
        report["hidden_sha256"] = "wrong"
        probe.save(path, report)
        with self.assertRaisesRegex(ValueError, "final-hidden"):
            probe.completed_run(state, "mac", "capture")

    def test_midpoint_worker_cannot_inherit_candidate_operator(self):
        probe.save(self.root / "started.json", {"pid": replay.os.getppid()})
        with patch.object(replay, "frozen", return_value={"midpoint": {"phase": "cross"}}), \
                patch.object(replay.os, "getpriority", return_value=10), \
                self.assertRaisesRegex(ValueError, "fixed pass"):
            replay.worker(self.root, "candidate")

    def test_export_keeps_verified_receipts_and_omits_large_parent_inputs(self):
        capture, _ = self.run_receipt("mac", "capture")
        own, _ = self.run_receipt("mac", "self", capture)
        protocol = {"midpoint": {"phase": "self", "source": capture.name}, "midpoint_root": str(self.root)}
        output = self.root / "bundle"
        with patch.object(replay, "frozen", return_value=protocol):
            replay.export_midpoint_cuda(own, output)
            with self.assertRaisesRegex(ValueError, "exists"):
                replay.export_midpoint_cuda(own, output)
        self.assertFalse((output / "inputs").exists())
        self.assertEqual(len(list((output / "probe").iterdir())), 8)
        probe.completed_run(output / "prerequisites/mac-capture", "mac", "capture")
        probe.completed_run(output / "prerequisites/mac-self", "mac", "self")
        for name, digest in json.loads((output / "source-hashes.json").read_text()).items():
            self.assertEqual(probe.sha256(output / "probe" / name), digest)

    def test_cuda_prepare_uses_completed_gates_and_refuses_overwrite(self):
        mac_capture, mac_self, cuda_capture, cuda_self = self.native_runs()
        for source, target in ((mac_capture, "prerequisites/mac-capture"), (mac_self, "prerequisites/mac-self"),
                               (cuda_capture, "results/midpoint-capture"), (cuda_self, "results/midpoint-self")):
            shutil.copytree(source, self.root / target)
        for phase in ("capture", "self", "cross"):
            config = probe.prepare_cuda(self.root, phase)
            path = self.root / f"midpoint-{phase}.json"
            self.assertEqual(json.loads(path.read_text()), config)
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(), probe.sha256(path))
            probe.validate_config(self.root, config, "cuda")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                probe.prepare_cuda(self.root, phase)

    def test_cpu_analysis_preserves_ties_and_compares_both_cross_directions(self):
        self.weight = torch.zeros((151936, 4), dtype=torch.bfloat16)
        self.logits = {name: torch.full((151936,), -20, dtype=torch.bfloat16) for name in ("mac", "cuda")}
        self.logits["mac"][151645], self.logits["mac"][3821] = 19.5, 19.25
        self.logits["cuda"][151645], self.logits["cuda"][3821] = 19.375, 19.375
        with patch.multiple(probe, VOCAB=151936, WEIGHT_SHA=probe.torch_digest(self.weight),
                            RAW_SHA={name: probe.torch_digest(value) for name, value in self.logits.items()}):
            mac_capture, mac_self, cuda_capture, cuda_self = self.native_runs()
            hidden, logits = self.hidden.copy(), self.logits.copy()
            self.hidden["mac"], self.logits["mac"] = hidden["cuda"].reshape(1, 1, 4), logits["cuda"]
            mac_cross, _ = self.run_receipt("mac", "cross", cuda_capture, mac_self, cuda_self)
            self.hidden["cuda"], self.logits["cuda"] = hidden["mac"].reshape(1, 4), logits["mac"]
            cuda_cross, _ = self.run_receipt("cuda", "cross", mac_capture, mac_self, cuda_self)
            result = probe.analyze(mac_self, cuda_self, mac_cross, cuda_cross)
            self.assertEqual(result["table"]["mac_self"]["argmax"], 151645)
            self.assertEqual(result["table"]["cuda_self"]["tie_count"], 2)
            self.assertEqual(result["table"]["mac_cross"]["eos_margin"], 0)
            self.assertEqual(result["table"]["cuda_cross"]["eos_margin"], 0.25)
            self.assertEqual(result["differences"]["mac_cross"]["cuda_self"]["logits"]["different_elements"], 0)
            self.assertEqual(result["differences"]["cuda_cross"]["mac_self"]["hidden"]["different_elements"], 0)


if __name__ == "__main__":
    unittest.main()
