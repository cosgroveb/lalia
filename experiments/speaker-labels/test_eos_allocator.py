import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

import eos_allocator as probe


class Cache:
    offset = 0
    keys = values = None

    @property
    def state(self):
        return self.keys, self.values


class EosAllocatorTests(unittest.TestCase):
    def test_historical_source_pins_use_measured_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            snapshot = state / "source-snapshot"
            snapshot.mkdir()
            current = state / "current.py"
            current.write_text("new code")
            archived = snapshot / "current.py"
            archived.write_text("measured code")
            expected = probe.sha256(archived)
            probe.save(state / "protocol.json", {"hashes": {str(current): expected}})
            probe.save(snapshot / "manifest.json", {"current.py": {"source": str(current), "sha256": expected,
                                                                   "in_frozen_protocol": True}})
            hashes = probe.historical_hashes(state)
            self.assertEqual(hashes, {str(archived): expected})
            probe.frozen.verify(hashes)
            archived.write_text("modified evidence")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                probe.frozen.verify(hashes)

    def test_schedule_is_exactly_eight_full_original_chunks(self):
        self.assertEqual(probe.chunks(), [(i * 2048, (i + 1) * 2048) for i in range(8)])
        self.assertEqual(probe.chunks()[-1], (14336, 16384))

    def test_kv_logical_and_allocated_bytes_remain_distinct(self):
        cache = Cache()
        self.assertEqual(probe.kv_bytes([cache]), {"kv_logical_bytes": 0, "kv_allocated_bytes": 0})
        cache.offset = 5
        cache.keys = cache.values = mx.zeros((1, 2, 8, 4), dtype=mx.bfloat16)
        self.assertEqual(probe.kv_bytes([cache]), {"kv_logical_bytes": 160, "kv_allocated_bytes": 256})

    def test_b_only_adds_clear_after_each_chunk_and_terminal_clear_has_no_forward(self):
        results = []
        for policy in (False, True):
            events = []
            model = Mock()
            model.make_cache = lambda: [Cache()]

            def forward(inputs_embeds, cache):
                events.append(("forward", inputs_embeds.shape[1]))
                cache[0].offset += inputs_embeds.shape[1]
                cache[0].keys = cache[0].values = mx.zeros((1, 1, cache[0].offset, 1))
                return inputs_embeds + 1

            model.model.language_model = Mock(side_effect=forward)
            clear = mx.clear_cache

            def tracked_clear():
                events.append(("clear", None))
                clear()

            with tempfile.TemporaryDirectory() as temporary, patch.object(mx, "clear_cache", tracked_clear):
                result = probe.prefix_pass(model, mx.zeros((probe.CONSUMED + 20, 2)), Path(temporary), policy)
            expected = ([("forward", 2048), ("clear", None)] * 8 if policy else [("forward", 2048)] * 8)
            self.assertEqual(events, expected + [("clear", None), ("clear", None)])
            before, after = result["measurements"][-2:]
            self.assertEqual((before["event"], after["event"]), ("before_terminal_clear", "after_terminal_clear"))
            self.assertEqual(before["kv_logical_bytes"], after["kv_logical_bytes"])
            self.assertEqual(before["kv_allocated_bytes"], after["kv_allocated_bytes"])
            self.assertEqual(result["consumed_tokens"], 16384)
            results.append(result)
        self.assertEqual(results[0]["final_hidden_float32_sha256"], results[1]["final_hidden_float32_sha256"])

    def test_mismatched_inputs_or_hidden_states_reject_comparison(self):
        inputs = {key: "same" for key in probe.INPUT_KEYS}
        left = {"inputs": inputs, "final_hidden_float32_sha256": "equal"}
        self.assertTrue(probe.compare(left, left)["final_hidden_equal"])
        for key in probe.INPUT_KEYS:
            with self.assertRaisesRegex(ValueError, "Frozen input"):
                probe.compare(left, {**left, "inputs": {**inputs, key: "changed"}})
        with self.assertRaisesRegex(ValueError, "Final hidden"):
            probe.compare(left, {**left, "final_hidden_float32_sha256": "changed"})

    def test_bf16_artifact_round_trip_preserves_digest_dtype_and_shape(self):
        embeddings = mx.array(np.arange(36, dtype=np.float32).reshape(12, 3)).astype(mx.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.safetensors"
            mx.save_safetensors(str(path), {"embeddings": embeddings})
            restored = mx.load(str(path))["embeddings"]
            self.assertEqual(restored.dtype, embeddings.dtype)
            self.assertEqual(restored.shape, embeddings.shape)
            self.assertEqual(probe.tensor_digest(restored), probe.tensor_digest(embeddings))

    def test_worker_failure_stops_before_b_and_reaps_children(self):
        launched = []
        launch = probe.Guard.launch

        def fake_launch(guard, command, log, **kwargs):
            code = "import time; time.sleep(30)" if not launched else "raise SystemExit(2)"
            child = launch(guard, [sys.executable, "-c", code], log)
            launched.append(child)
            return child

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            probe.save(state / "protocol.json", {"hashes": {}})
            with patch.object(probe, "power", return_value="test"), patch.object(probe.Guard, "launch", fake_launch):
                with self.assertRaisesRegex(RuntimeError, "exited 2"):
                    probe.execute(state)
            self.assertEqual(len(launched), 2)
            self.assertTrue(all(child.poll() is not None for child in launched))
            self.assertFalse((state / "B").exists())
            self.assertTrue(json.loads((state / "result.json").read_text())["frozen_files_unchanged"])

    def test_final_verification_failure_marks_result_failed_and_preserves_worker_error(self):
        for worker_fails in (False, True):
            with self.subTest(worker_fails=worker_fails), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                probe.save(state / "protocol.json", {"hashes": {}})
                guard = Mock(started=probe.time.monotonic(), samples=[])

                def launch(command, log, **kwargs):
                    if command[0] != "caffeinate":
                        if worker_fails:
                            raise RuntimeError("synthetic worker failure")
                        probe.save(log.parent / "result.json", {"inputs": {key: "same" for key in probe.INPUT_KEYS},
                                                               "final_hidden_float32_sha256": "same"})
                    return Mock()

                guard.launch.side_effect = launch
                with patch.object(probe, "Guard", return_value=guard), patch.object(probe, "power", return_value="test"), patch.object(probe.frozen, "verify", side_effect=[None, ValueError("synthetic frozen-file change")]):
                    with self.assertRaisesRegex(ValueError, "synthetic frozen-file change"):
                        probe.execute(state)
                guard.close.assert_called_once_with()
                result = json.loads((state / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["frozen_files_unchanged"])
                self.assertEqual(result["verification_error"], "ValueError: synthetic frozen-file change")
                expected = "RuntimeError: synthetic worker failure" if worker_fails else result["verification_error"]
                self.assertEqual(result["error"], expected)


if __name__ == "__main__":
    unittest.main()
