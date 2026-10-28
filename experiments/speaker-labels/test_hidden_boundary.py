from contextlib import nullcontext
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

import hidden_boundary as probe
import residual_rope as replay
from cuda import supervise as cuda_supervise


class HiddenBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mx.set_default_device, mx.default_device())
        mx.set_default_device(mx.cpu)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) / 16
        self.mac = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.bfloat16)
        self.cuda = torch.tensor([[4, 3, 2, 1]], dtype=torch.bfloat16)
        self.mac_logits = torch.nn.functional.linear(self.mac, self.weight)[0, 0]
        self.cuda_logits = torch.nn.functional.linear(self.cuda, self.weight)[0]
        self.enterContext(patch.multiple(probe, WIDTH=4, VOCAB=5, SHAPES={"mac": (1, 1, 4), "cuda": (1, 4)},
                                         WEIGHT_SHA=probe.torch_digest(self.weight),
                                         RAW_SHA={"mac": probe.torch_digest(self.mac_logits),
                                                  "cuda": probe.torch_digest(self.cuda_logits)}))

    def mlx(self, value):
        return mx.array(value.view(torch.uint16).numpy()).view(mx.bfloat16)

    def capture(self):
        embedding = nn.Embedding.__new__(nn.Embedding)
        nn.Module.__init__(embedding)
        embedding.weight = self.mlx(self.weight)
        return probe.capture_mlx(self.mlx(self.mac), self.mlx(self.mac_logits), embedding, self.root)

    def test_capture_keeps_native_shape_and_exact_bf16_bytes(self):
        report = self.capture()
        values, readback = probe.read_torch(self.root, "mac")
        self.assertEqual(readback, report)
        self.assertTrue(torch.equal(values["hidden"], self.mac))
        self.assertTrue(torch.equal(values["logits"], self.mac_logits))
        self.assertEqual(report["hidden_shape"], [1, 1, 4])
        with self.assertRaisesRegex(ValueError, "shape"):
            probe.require_tensor(probe.mlx_digest, self.mlx(self.mac).reshape(1, 4), (1, 1, 4))

    def test_capture_requires_full_original_logits_and_weight_bytes(self):
        embedding = nn.Embedding.__new__(nn.Embedding)
        nn.Module.__init__(embedding)
        embedding.weight = self.mlx(self.weight)
        for logits, weight in ((self.mac_logits + 1, self.weight), (self.mac_logits, self.weight + 1)):
            embedding.weight = self.mlx(weight)
            with self.subTest(logits=logits), self.assertRaisesRegex(ValueError, "bytes differ"):
                probe.capture_mlx(self.mlx(self.mac), self.mlx(logits), embedding, self.root)
        self.assertFalse((self.root / "mac-head.json").exists())

    def test_capture_file_drift_rejects_before_loading(self):
        self.capture()
        path = self.root / "mac-head.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "file changed"):
            probe.read_torch(self.root, "mac")

    def test_resealed_wrong_tensor_still_fails_actual_digest(self):
        report = self.capture()
        path = self.root / "mac-head.safetensors"
        save_file({"hidden": self.mac + 1, "logits": self.mac_logits}, path)
        report["file_sha256"] = probe.sha256(path)
        probe.save(self.root / "mac-head.json", report)
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            probe.read_torch(self.root, "mac")

    def test_cuda_cross_projection_preserves_rank_and_gates_own_result_first(self):
        self.capture()
        output = self.root / "cuda"
        output.mkdir()
        calls = []
        original = torch.nn.functional.linear

        def linear(hidden, weight):
            calls.append(tuple(hidden.shape))
            return original(hidden, weight)

        with patch.object(torch.amp, "autocast", return_value=nullcontext()), \
                patch.object(torch.nn.functional, "linear", side_effect=linear):
            result = probe.torch_capture_heads(self.cuda, self.cuda_logits, self.weight, output, self.root)
        self.assertEqual(calls, [(1, 4), (1, 4)])
        self.assertFalse(result["identical_hidden"])
        self.assertEqual(result["self_sha256"], probe.RAW_SHA["cuda"])
        self.assertEqual(result["cross_sha256"], probe.RAW_SHA["mac"])

    def test_failed_cuda_self_projection_never_attempts_cross(self):
        self.capture()
        output = self.root / "cuda"
        output.mkdir()
        with patch.object(torch.amp, "autocast", return_value=nullcontext()), \
                patch.object(torch.nn.functional, "linear", return_value=self.cuda_logits[None] + 1) as project:
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                probe.torch_capture_heads(self.cuda, self.cuda_logits, self.weight, output, self.root)
        self.assertEqual(project.call_count, 1)
        self.assertFalse((output / "head-projections.safetensors").exists())

    def test_identical_hidden_states_skip_both_new_projections(self):
        self.capture()
        output = self.root / "cuda"
        output.mkdir()
        with patch.dict(probe.RAW_SHA, {"cuda": probe.RAW_SHA["mac"]}), \
                patch.object(torch.nn.functional, "linear") as project:
            result = probe.torch_capture_heads(self.mac.reshape(1, 4), self.mac_logits, self.weight, output, self.root)
        self.assertTrue(result["identical_hidden"])
        project.assert_not_called()

    def test_mlx_head_only_loads_embedding_and_uses_own_native_rank(self):
        self.capture()
        output = self.root / "cuda"
        output.mkdir()
        with patch.object(torch.amp, "autocast", return_value=nullcontext()):
            probe.torch_capture_heads(self.cuda, self.cuda_logits, self.weight, output, self.root)
        model_path = self.root / "model.safetensors"
        save_file({probe.WEIGHT_KEY: self.weight, "unused": torch.tensor([7])}, model_path)
        head = self.root / "head"
        head.mkdir()
        result = probe.mlx_heads(model_path, self.root, output, head)
        self.assertEqual(result["self_shape"], [1, 1, 4])
        self.assertEqual(result["cross_shape"], [1, 1, 4])
        self.assertEqual(result["self_sha256"], probe.RAW_SHA["mac"])
        self.assertEqual(result["cross_sha256"], probe.RAW_SHA["cuda"])

    def test_nonfinite_or_wrong_dtype_boundary_rejected(self):
        for value in (torch.full((1, 4), float("nan"), dtype=torch.bfloat16), torch.ones((1, 4))):
            with self.assertRaisesRegex(ValueError, "finite BF16"):
                probe.torch_digest(value)

    def test_head_prepare_requires_both_capture_paths_before_reading_sources(self):
        for boundary, mac, cuda in (("heads", None, self.root), ("heads", self.root, None),
                                    (None, self.root, None), ("capture", self.root, self.root)):
            with self.subTest(boundary=boundary, mac=mac, cuda=cuda), self.assertRaisesRegex(ValueError, "both completed"):
                replay.prepare(self.root / "new", boundary=boundary, capture_state=mac, cuda_state=cuda)

    def test_head_protocol_rejects_decoder_worker_before_import_or_load(self):
        probe.save(self.root / "started.json", {"pid": replay.os.getppid()})
        with patch.object(replay, "frozen", return_value={"boundary": "heads"}), \
                patch.object(replay.os, "getpriority", return_value=10), \
                self.assertRaisesRegex(ValueError, "only the head worker"):
            replay.worker(self.root, "baseline")

    def archive(self):
        root = self.root / "archive"
        inputs = root / "inputs"
        inputs.mkdir(parents=True)
        files = ("combined-embeddings.safetensors", "config.json", "model-00000-of-00001.safetensors")
        for name in files:
            (inputs / name).write_text(name)
        manifest = {"model_revision": replay.MODEL_REVISION, "embedding_float32_sha256": replay.CONTENT_SHA,
                    "transformers_sources": {}, "files": {name: probe.sha256(inputs / name) for name in files}}
        probe.save(inputs / "manifest.json", manifest)
        scorer = root / "probe/score_decoder.py"
        scorer.parent.mkdir()
        scorer.write_text("TOTAL = 101135\nCHUNK = 4096\n")
        result = {"consumed": replay.TOTAL, "cache_tokens": replay.TOTAL, "chunk": replay.CHUNK,
                  "final_position": replay.TOTAL - 1, "probe_sha256": probe.sha256(scorer),
                  "fp32_rope": False, "rope_dtype": "torch.bfloat16", "raw_bf16_sha256": probe.RAW_SHA["cuda"],
                  "logits_file_sha256": replay.CUDA_SHA, "model_revision": replay.MODEL_REVISION}
        result_path = root / "execution/remote/results/score/model/result.json"
        result_path.parent.mkdir(parents=True)
        probe.save(result_path, result)
        protocol = {"consumed_tokens": replay.TOTAL, "chunk_tokens": replay.CHUNK, "final_position": replay.TOTAL - 1,
                    "hashes": {"/archive/cuda/score_decoder.py": probe.sha256(scorer)}}
        historical = root / "historical"
        historical.mkdir()
        probe.save(historical / "protocol.json", protocol)
        pins = {name: probe.sha256(root / name) for name in replay.CUDA_ARCHIVE_PINS}
        self.enterContext(patch.multiple(replay, CUDA_INPUTS=inputs, CUDA_ARCHIVE_PINS=pins,
                                         HISTORICAL=historical, HISTORICAL_SHA=probe.sha256(historical / "protocol.json")))
        return manifest, result_path, result

    def test_archived_schema_without_profile_binds_executed_source_and_schedule(self):
        original, _, _ = self.archive()
        manifest, hashes = replay.archived_cuda_inputs()
        self.assertNotIn("score_profile", original)
        self.assertEqual(manifest, original)
        self.assertIn(str(replay.HISTORICAL / "protocol.json"), hashes)
        for name in replay.CUDA_ARCHIVE_PINS:
            self.assertIn(str(replay.CUDA_INPUTS.parent / name), hashes)

    def test_archived_provenance_rejects_resealed_wrong_schedule_or_source(self):
        _, path, original = self.archive()
        for field, value in (("chunk", 512), ("probe_sha256", "wrong"), ("model_revision", "wrong"), ("fp32_rope", True)):
            with self.subTest(field=field):
                probe.save(path, dict(original, **{field: value}))
                replay.CUDA_ARCHIVE_PINS[str(path.relative_to(replay.CUDA_INPUTS.parent))] = probe.sha256(path)
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    replay.archived_cuda_inputs()

    def test_boundary_export_adds_profile_and_passes_supervisor_input_validation(self):
        original, _, _ = self.archive()
        self.capture()
        baseline = self.root / "baseline"
        baseline.mkdir()
        for name in ("mac-head.json", "mac-head.safetensors"):
            shutil.copy2(self.root / name, baseline / name)
        output = self.root / "bundle"
        with patch.object(replay, "completed_capture"):
            replay.export_hidden_cuda(self.root, output)
        manifest = json.loads((output / "inputs/manifest.json").read_text())
        self.assertEqual(manifest["score_profile"], "exact-prefix")
        self.assertIs(manifest["hidden_boundary"], True)
        self.assertEqual(json.loads((replay.CUDA_INPUTS / "manifest.json").read_text()), original)
        supervisor = cuda_supervise.Supervisor(output, "score", exact_prefix=True, hidden_boundary=True)
        supervisor.verify_boundary_inputs()
        del manifest["score_profile"]
        probe.save(output / "inputs/manifest.json", manifest)
        (output / "input-manifest.sha256").write_text(probe.sha256(output / "inputs/manifest.json") + "\n")
        with self.assertRaisesRegex(ValueError, "profile differs"):
            supervisor.verify_boundary_inputs()


class BoundaryAnalysisTests(unittest.TestCase):
    def test_table_uses_full_vectors_and_one_analysis_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mac, cuda, head = (root / name for name in ("mac", "cuda", "head"))
            for directory in (mac, cuda, head):
                directory.mkdir()
            mm = torch.full((probe.VOCAB,), -2, dtype=torch.bfloat16)
            mm[151645], mm[3821] = 19.5, 19.25
            cc = mm.clone()
            cc[151645], cc[3821] = 19.375, 19.375
            mc, cm = cc.clone(), mm.clone()
            digests = {"mac": probe.torch_digest(mm), "cuda": probe.torch_digest(cc)}
            with patch.dict(probe.RAW_SHA, digests):
                for backend, directory, hidden, logits in (
                        ("mac", mac, torch.zeros(probe.SHAPES["mac"], dtype=torch.bfloat16), mm),
                        ("cuda", cuda, torch.ones(probe.SHAPES["cuda"], dtype=torch.bfloat16), cc)):
                    path = directory / f"{backend}-head.safetensors"
                    save_file({"hidden": hidden, "logits": logits}, path)
                    probe.save(directory / f"{backend}-head.json", {"backend": backend, "dtype": "BF16",
                               "hidden_shape": list(hidden.shape), "hidden_sha256": probe.torch_digest(hidden),
                               "logits_sha256": digests[backend], "weight_shape": [probe.VOCAB, probe.WIDTH],
                               "weight_sha256": probe.WEIGHT_SHA, "file_sha256": probe.sha256(path)})
                save_file({"self_logits": cc, "cross_logits": cm}, cuda / "head-projections.safetensors")
                save_file({"logits": mm, "cross_logits": mc}, head / "scores.safetensors")
                probe.save(cuda / "result.json", {"boundary": {"identical_hidden": False,
                           "cross_sha256": probe.torch_digest(cm),
                           "projections_sha256": probe.sha256(cuda / "head-projections.safetensors")}})
                probe.save(head / "result.json", {"status": "complete", "cross_sha256": probe.torch_digest(mc),
                           "scores_sha256": probe.sha256(head / "scores.safetensors")})
                result = probe.analyze(mac, cuda, head)
                self.assertEqual(result["table"]["mac_head_mac_hidden"]["eos_margin"], 0.25)
                self.assertEqual(result["table"]["cuda_head_cuda_hidden"]["argmax"], 3821)
                self.assertEqual(result["table"]["cuda_head_cuda_hidden"]["tie_count"], 2)
                self.assertEqual(result["hidden_difference"]["different_elements"], 1024)
                self.assertEqual(result["anchored_differences"]["head_at_cuda_hidden"]["max_absolute_difference"], 0)
                self.assertEqual(result["anchored_differences"]["head_at_mac_hidden"]["max_absolute_difference"], 0)
                self.assertIn("not native", result["normalization"])


if __name__ == "__main__":
    unittest.main()
