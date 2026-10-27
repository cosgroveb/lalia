import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

import eos_rope as probe
import short_windows

torch.set_num_threads(2)


class EosRoPETests(unittest.TestCase):
    def test_operator_pins_do_not_depend_on_wheel_file_metadata(self):
        with patch.object(probe.importlib.metadata, "distribution", return_value=Mock(files=None)):
            operators = probe.installed_operators()
        self.assertTrue(any(path.suffix == ".so" for path in operators["mlx"]))
        self.assertTrue(any(path.name == "libmlx.dylib" for path in operators["mlx"]))
        self.assertTrue(any(path.suffix == ".metallib" for path in operators["mlx"]))
        self.assertTrue(any(path.name == "libtorch_cpu.dylib" for path in operators["torch"]))
        self.assertEqual(len(operators["python_sources"]), 3)
        self.assertTrue(all(path.is_file() for group in operators.values() for path in group))
        original = Path.is_file
        with patch.object(Path, "is_file", lambda path: path.suffix != ".metallib" and original(path)):
            with self.assertRaisesRegex(ValueError, "Missing required.*binaries"):
                probe.installed_operators()
        with patch.object(Path, "is_file", lambda path: path.name != "modeling_rope_utils.py" and original(path)):
            with self.assertRaisesRegex(ValueError, "Missing required.*Python sources"):
                probe.installed_operators()

    def test_prefix_assembly_preserves_prompt_and_saved_token_order(self):
        embed = nn.Embedding(10, 2)
        embed.weight = mx.array(np.arange(20).reshape(10, 2), dtype=mx.float32)
        prompt = mx.array([[100., 101.], [102., 103.]])
        result = probe.assemble(mx.array([1, 2]), prompt, [7, 8, 9], embed, 2, 3)
        np.testing.assert_array_equal(np.array(result), [[100, 101], [102, 103], [14, 15], [16, 17], [18, 19]])
        for prefix in ([7, 8], [7, True, 9], [7, 8, 10], [7, 8, probe.EOS]):
            with self.assertRaises(ValueError):
                probe.assemble(mx.array([1, 2]), prompt, prefix, embed, 2, 3)
        with self.assertRaises(ValueError):
            probe.assemble(mx.array([1]), prompt, [7, 8, 9], embed, 2, 3)

    def test_chunks_cover_exact_consumed_prefix_then_position_88813(self):
        spans = list(probe.chunks(probe.TOTAL))
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1], (88813, 88814))
        self.assertEqual(sum(end - start for start, end in spans), 88814)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertTrue(all(0 < end - start <= probe.CHUNK for start, end in spans))

    def test_reference_never_runs_when_native_control_differs(self):
        reference = Mock(return_value={"sampler_id": 5})
        result = probe.compare({"sampler_id": 151643, "raw_argmax_id": probe.EOS}, reference)
        reference.assert_not_called()
        self.assertEqual(result["status"], "native_control_not_reproduced")
        result = probe.compare({"sampler_id": probe.EOS}, reference)
        reference.assert_called_once_with()
        self.assertEqual(result["reference"], {"sampler_id": 5})

    def test_reference_matches_torch_eager_at_long_positions_cpu_and_gpu(self):
        values = np.random.default_rng(1).normal(size=(1, 2, 4, 128)).astype(np.float32)
        original_device = mx.default_device()
        try:
            for device in (mx.cpu, mx.gpu):
                mx.set_default_device(device)
                rope = probe.ReferenceRoPE(88818)
                for dtype in ("float32", "bfloat16"):
                    q = torch.from_numpy(values).to(getattr(torch, dtype))
                    for offset in (0, 71687, 88813):
                        inverse = 1 / (1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
                        positions = torch.arange(offset, offset + 4, dtype=torch.float32)
                        angles = (inverse[None, :, None] @ positions[None, None, :]).transpose(1, 2)
                        doubled = torch.cat((angles, angles), dim=-1)
                        cos = doubled.cos().to(q.dtype)[:, None, :, :]
                        sin = doubled.sin().to(q.dtype)[:, None, :, :]
                        expected = q * cos + torch.cat((-q[..., 64:], q[..., :64]), dim=-1) * sin
                        actual = rope(mx.array(values).astype(getattr(mx, dtype)), offset=offset)
                        with self.subTest(device=str(device), dtype=dtype, offset=offset):
                            np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected.float().numpy())
        finally:
            mx.set_default_device(original_device)
            mx.clear_cache()

    def test_probabilities_normalize_after_raw_bf16_logits_cast_to_fp32(self):
        logits = mx.full((151646,), -100., dtype=mx.bfloat16)
        logits[probe.EOS], logits[42474] = 16., 15.875
        result = probe.scores(logits)
        self.assertEqual(result["sampler_id"], probe.EOS)
        self.assertEqual(result["raw_argmax_id"], probe.EOS)
        self.assertEqual(result["eos_rank"], 1)
        self.assertEqual(result["best_non_eos_id"], 42474)
        self.assertEqual(result["best_non_eos_rank"], 2)
        self.assertEqual(result["eos_minus_best_non_eos_logit"], 0.125)
        self.assertAlmostEqual(result["probability_sum"], 1, places=6)
        self.assertAlmostEqual(result["eos_probability"], 1 / (1 + np.exp(-.125)), places=6)

    def test_native_gate_uses_original_dtype_sampler_when_normalization_ties(self):
        logits = mx.zeros((151646,), dtype=mx.bfloat16)
        logits[probe.EOS] = .001
        result = probe.scores(logits)
        self.assertEqual(result["raw_argmax_id"], probe.EOS)
        self.assertEqual(result["sampler_id"], 0)
        reference = Mock()
        self.assertEqual(probe.compare(result, reference)["status"], "native_control_not_reproduced")
        reference.assert_not_called()

    def test_reference_cache_is_released_before_second_pass(self):
        class Cache:
            offset = 0
            state = ()

        live = []
        model = Mock()

        def make_cache():
            import weakref
            self.assertTrue(all(reference() is None for reference in live))
            cache = Cache()
            live.append(weakref.ref(cache))
            return [cache]

        def forward(inputs_embeds, cache):
            cache[0].offset += inputs_embeds.shape[1]
            return inputs_embeds

        model.make_cache = make_cache
        model.model.language_model = Mock(side_effect=forward)
        model.lm_head = lambda values: mx.zeros((1, 1, 151646))
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            for phase in ("native", "reference"):
                probe.token_pass(model, mx.zeros((3, 2)), state, phase)
        self.assertTrue(all(reference() is None for reference in live))

    def test_allocator_clear_follows_evaluation_before_next_chunk(self):
        class Cache:
            offset = 0
            state = ()

        events = []
        model = Mock()
        model.make_cache = lambda: [Cache()]

        def forward(inputs_embeds, cache):
            events.append(("forward", cache[0].offset))
            cache[0].offset += inputs_embeds.shape[1]
            return inputs_embeds

        evaluate, clear = mx.eval, mx.clear_cache

        def tracked_eval(*values):
            events.append(("eval", None))
            return evaluate(*values)

        def tracked_clear():
            events.append(("clear", None))
            clear()

        model.model.language_model = Mock(side_effect=forward)
        model.lm_head = lambda values: mx.zeros((1, 1, 151646))
        with tempfile.TemporaryDirectory() as temporary, patch.object(mx, "eval", tracked_eval), patch.object(mx, "clear_cache", tracked_clear):
            probe.token_pass(model, mx.zeros((4097, 2)), Path(temporary), "native")
        self.assertEqual(events[:9], [("forward", 0), ("eval", None), ("clear", None),
                                     ("forward", 2048), ("eval", None), ("clear", None),
                                     ("forward", 4096), ("eval", None), ("clear", None)])

    def test_worker_launch_failure_reaps_awake_child(self):
        launched = []
        launch = probe.Guard.launch

        def fake_launch(guard, command, log, **kwargs):
            if launched:
                raise RuntimeError("synthetic launch failure")
            child = launch(guard, [sys.executable, "-c", "import time; time.sleep(30)"], log)
            launched.append(child)
            return child

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            probe.save(state / "protocol.json", {"hashes": {}, "limits": {"sampled_footprint_bytes": 32 * 1024**3}})
            with patch.object(probe, "power", return_value="test"), patch.object(probe.Guard, "launch", fake_launch):
                with self.assertRaisesRegex(RuntimeError, "synthetic launch failure"):
                    probe.execute(state)
            self.assertIsNotNone(launched[0].poll())
            result = json.loads((state / "result.json").read_text())
            self.assertTrue(result["frozen_files_unchanged"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(json.loads((state / "cleanup.json").read_text())["children"]), 1)

    def test_final_verification_failure_marks_result_failed_and_preserves_worker_error(self):
        for returncode in (0, 2):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                probe.save(state / "protocol.json", {"hashes": {}, "limits": {"sampled_footprint_bytes": 32 * 1024**3}})
                probe.save(state / "comparison.json", {"status": "complete"})
                child = Mock(returncode=returncode)
                child.poll.return_value = returncode
                guard = Mock(started=probe.time.monotonic(), samples=[])
                guard.launch.return_value = child
                with patch.object(probe, "Guard", return_value=guard), patch.object(probe, "power", return_value="test"), patch.object(probe, "verify", side_effect=[None, ValueError("synthetic frozen-file change")]):
                    with self.assertRaisesRegex(ValueError, "synthetic frozen-file change"):
                        probe.execute(state)
                guard.close.assert_called_once_with()
                result = json.loads((state / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["frozen_files_unchanged"])
                self.assertEqual(result["verification_error"], "ValueError: synthetic frozen-file change")
                expected = "RuntimeError: Worker exited 2; see worker.log" if returncode else result["verification_error"]
                self.assertEqual(result["error"], expected)

    def test_footprint_limit_preserves_existing_default(self):
        child = Mock()
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, patch.object(short_windows, "footprint", return_value=20 * 1024**3), patch.object(short_windows, "power"):
            guard = probe.Guard(Path(temporary))
            guard.children = [child]
            with self.assertRaisesRegex(MemoryError, "16GiB"):
                guard.check()
            probe_guard = probe.Guard(Path(temporary), footprint_limit=32 * 1024**3)
            probe_guard.children = [child]
            probe_guard.check()


if __name__ == "__main__":
    unittest.main()
