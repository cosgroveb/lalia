import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import mlx.core as mx

import exact_prefix as probe
import eos_rope


class ExactPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def evidence(self):
        observed = {"text_sha256": "a" * 64, "output_tokens": probe.GENERATED, "eos_id": probe.EOS}
        status = {"status": "complete", "reproduces_previous": True,
                  "observed": observed, "expected": dict(observed)}
        tokens = [42] * probe.GENERATED + [probe.EOS]
        eos = {"token_id": probe.EOS, "output_tokens": probe.GENERATED,
               "yielded_tokens": probe.GENERATED + 1, "argmax_id": probe.EOS}
        return status, tokens, eos

    def test_exact_capture_requires_complete_matching_observation(self):
        status, tokens, eos = self.evidence()
        probe.validate_trace(status, tokens, eos)
        for field, value in (("status", "incomplete"), ("reproduces_previous", False)):
            changed = {**status, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "complete exact reproduction"):
                probe.validate_trace(changed, tokens, eos)
        changed = copy.deepcopy(status)
        changed["expected"]["text_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "differs from previous"):
            probe.validate_trace(changed, tokens, eos)

    def test_captured_ids_exclude_eos_until_final_and_have_exact_length(self):
        status, tokens, eos = self.evidence()
        for changed in (tokens[:-1], [True] + tokens[1:], [probe.EOS] + tokens[1:],
                        [probe.VOCAB] + tokens[1:], tokens[:-1] + [151643]):
            with self.subTest(first=changed[0], length=len(changed)), self.assertRaisesRegex(ValueError, "Invalid captured"):
                probe.validate_trace(status, changed, eos)

    def test_eos_vector_must_be_paired_with_final_yield(self):
        status, tokens, eos = self.evidence()
        for field, value in (("output_tokens", probe.GENERATED + 1),
                             ("yielded_tokens", probe.GENERATED), ("argmax_id", 42)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "pairing or sampler"):
                probe.validate_trace(status, tokens, {**eos, field: value})

    def test_whole_prefix_batches_preserve_final_position_once(self):
        spans = list(probe.chunks(probe.TOTAL, probe.CHUNK))
        self.assertEqual(spans[-1], (101134, 101135))
        self.assertEqual(sum(end - start for start, end in spans), 101135)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertEqual(spans[0], (0, 4096))

    def test_replay_distinguishes_selection_from_bitwise_reproduction(self):
        logits = mx.full((probe.VOCAB,), -10., dtype=mx.bfloat16)
        logits[probe.EOS], logits[42] = 2., 1.
        captured = (logits[None] - mx.logsumexp(logits[None], keepdims=True)).squeeze(0)
        with patch.object(mx, "logsumexp", wraps=mx.logsumexp) as normalize:
            result, _ = probe.compare_vectors(logits, captured)
        self.assertEqual(normalize.call_args.args[0].shape, (1, probe.VOCAB))
        self.assertEqual(normalize.call_args.kwargs, {"keepdims": True})
        self.assertEqual(result["status"], "selection_reproduced")
        self.assertTrue(result["logprobs_bitwise_equal"])
        changed = mx.array(logits)
        changed[100] = -9.
        result, _ = probe.compare_vectors(changed, captured)
        self.assertEqual(result["status"], "selection_reproduced")
        self.assertFalse(result["logprobs_bitwise_equal"])
        changed[42] = 3.
        result, _ = probe.compare_vectors(changed, captured)
        self.assertEqual(result["status"], "execution_shape_mismatch")

    def test_nonfinite_or_non_eos_captured_vector_is_rejected(self):
        logits = mx.zeros((probe.VOCAB,), dtype=mx.bfloat16)
        logits[probe.EOS] = 2.
        with self.assertRaisesRegex(ValueError, "dtypes differ"):
            probe.compare_vectors(logits, logits.astype(mx.float32))
        with self.assertRaisesRegex(ValueError, "does not select"):
            probe.compare_vectors(logits, mx.zeros_like(logits))
        logits[0] = mx.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            probe.vector_summary(logits)

    def test_replay_reuses_candidate_class_and_measured_denominators(self):
        self.assertIs(probe.NativeBF16RoPE, eos_rope.NativeBF16RoPE)
        rope = probe.NativeBF16RoPE(131072)
        self.assertEqual(rope.frequencies["denominator_sha256"], probe.DENOMINATOR_SHA)

    def test_supervisor_closes_children_and_saves_worker_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            (state / "protocol.json").write_text("{}")
            (state / "protocol.sha256").write_text("unused")
            child = types.SimpleNamespace(poll=lambda: 7, wait=lambda: None, returncode=7)
            guard = types.SimpleNamespace(started=probe.time.monotonic(), samples=[],
                                          launch=lambda *a, **kw: child, close=unittest.mock.Mock())
            protocol = {"hashes": {}, "limits": {"sampled_footprint_bytes": 64 * 1024**3}}
            with patch.object(probe, "frozen", return_value=protocol), patch.object(probe, "verify"), \
                    patch.object(probe, "Guard", return_value=guard), patch.object(probe, "power", return_value="AC"), \
                    patch.object(probe.signal, "signal"), patch.object(probe.signal, "alarm"):
                with self.assertRaisesRegex(RuntimeError, "Worker exited 7"):
                    probe.execute(state)
            guard.close.assert_called_once()
            result = json.loads((state / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["frozen_files_unchanged"])

    def test_cuda_export_rejects_selection_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = Path(temp), Path(temp) / "cuda"
            (state / "result.json").write_text(json.dumps({"status": "execution_shape_mismatch",
                                                          "frozen_files_unchanged": True}))
            with patch.object(probe, "frozen", return_value={}):
                with self.assertRaisesRegex(ValueError, "validated MLX selection"):
                    probe.export_cuda(state, output)
            self.assertFalse(output.exists())

    def test_cuda_export_reuses_shared_sources_with_explicit_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            trace, model, output = state / "trace", state / "model", state / "cuda"
            trace.mkdir()
            model.mkdir()
            embeddings = mx.zeros((2, 4), dtype=mx.bfloat16)
            mx.save_safetensors(str(trace / "combined-embeddings.safetensors"), {"inputs_embeds": embeddings})
            (trace / "status.json").write_text(json.dumps({"combined_embeddings": {"float32_content_sha256": "digest"}}))
            (state / "scores.safetensors").write_bytes(b"scores")
            (state / "result.json").write_text(json.dumps({"status": "selection_reproduced",
                "frozen_files_unchanged": True, "scores_sha256": probe.sha256(state / "scores.safetensors")}))
            manifest = json.loads((probe.SOURCE / "cuda/manifest.json").read_text())
            for name in ("config.json", "model-00000-of-00001.safetensors"):
                (model / name).write_bytes(b"model fixture")
            original_sha = probe.sha256
            def digest(path):
                return manifest["files"][path.name] if path.parent == model else original_sha(path)
            with patch.object(probe, "frozen", return_value={"trace": str(trace)}), \
                    patch.object(probe, "MODEL", model), patch.object(probe, "sha256", side_effect=digest):
                probe.export_cuda(state, output)
                with self.assertRaisesRegex(ValueError, "No overwrite"):
                    probe.export_cuda(state, output)
            exported = json.loads((output / "inputs/manifest.json").read_text())
            self.assertEqual(exported["score_profile"], "exact-prefix")
            self.assertEqual(exported["embedding_float32_sha256"], "digest")
            saved = mx.load(str(output / "inputs/combined-embeddings.safetensors"))["embeddings"]
            self.assertTrue(bool(mx.array_equal(saved, embeddings).item()))
            for name in ("score_decoder.py", "supervise.py", "smoke_numeric.py"):
                self.assertEqual((output / "probe" / name).read_bytes(), (probe.SOURCE / "cuda" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
