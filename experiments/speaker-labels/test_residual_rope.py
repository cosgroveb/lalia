import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import residual_rope as probe


def baseline():
    summary = {"argmax_id": probe.EOS, "dtype": "mlx.core.bfloat16", "eos_margin": 0.25,
               "eos_minus_3821": 0.25, "best_continuation_id": 3821}
    return {"consumed_tokens": probe.TOTAL, "cache_offsets": [probe.TOTAL] * 28,
            "chunk_tokens": probe.CHUNK, "final_position": probe.TOTAL - 1,
            "raw_logits": dict(summary, raw_sha256=probe.RAW_SHA),
            "replay": dict(summary, raw_sha256=probe.NORMALIZED_SHA), "scores_sha256": "f" * 64}


class ResidualRoPETests(unittest.TestCase):
    def test_cpu_vector_summary_preserves_native_tie_and_reports_whole_vector_difference(self):
        import mlx.core as mx

        previous = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            values = mx.full((151936,), -20, dtype=mx.bfloat16)
            values[probe.EOS], values[3821] = 19.375, 19.375
            summary = probe.summary(values)
            self.assertEqual((summary["argmax_id"], summary["tie_count"], summary["eos_minus_3821"]), (3821, 2, 0))
            baseline = mx.array(values)
            baseline[probe.EOS], baseline[3821] = 19.5, 19.25
            difference = probe.differences(values, baseline)
            self.assertEqual(difference["max_absolute_difference"], 0.125)
            self.assertAlmostEqual(difference["mean_absolute_difference"], 0.25 / 151936, places=11)
            values[0] = mx.nan
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                probe.summary(values)
        finally:
            mx.set_default_device(previous)

    def test_baseline_requires_both_full_vectors_not_only_eos(self):
        probe.require_baseline(baseline())
        for vector in ("raw_logits", "replay"):
            for field, value in (("raw_sha256", "a" * 64), ("argmax_id", 3821),
                                 ("eos_margin", 0.125), ("dtype", "mlx.core.float32"),
                                 ("eos_minus_3821", 0), ("best_continuation_id", 42)):
                changed = baseline()
                changed[vector][field] = value
                with self.subTest(vector=vector, field=field), self.assertRaisesRegex(ValueError, "exact historical vector"):
                    probe.require_baseline(changed)
        for field, value in (("consumed_tokens", probe.TOTAL - 1), ("cache_offsets", [probe.TOTAL] * 27),
                             ("chunk_tokens", 2048), ("final_position", probe.TOTAL)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cache/schedule"):
                probe.require_baseline(dict(baseline(), **{field: value}))

    def test_selects_whole_existing_operator_with_identical_inverse_bytes(self):
        native = types.SimpleNamespace(frequencies={"stored_sha256": probe.BF16_FREQUENCY_SHA256,
                                                   "denominator_sha256": probe.DENOMINATOR_SHA})
        reference = types.SimpleNamespace(frequencies={"stored_sha256": probe.BF16_FREQUENCY_SHA256})
        with patch.object(probe, "NativeBF16RoPE", return_value=native) as n, patch.object(probe, "ReferenceRoPE", return_value=reference) as r:
            self.assertIs(probe.select_operator("baseline"), native)
            n.assert_called_once_with(131072)
            r.assert_not_called()
            self.assertIs(probe.select_operator("candidate"), reference)
            r.assert_called_once_with(131072, bf16_frequencies=True)
            reference.frequencies["stored_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "inverse frequencies"):
                probe.select_operator("candidate")
            native.frequencies["denominator_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "denominators"):
                probe.select_operator("baseline")

    def test_frozen_inputs_sources_and_runtime_inventory_cannot_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            runtime = state / "runtime"
            runtime.mkdir()
            source, inputs = runtime / "operator.py", state / "embeddings"
            source.write_text("original operator")
            inputs.write_bytes(b"original input")
            protocol = {"hashes": {str(p): probe.sha256(p) for p in (source, inputs)},
                        "runtime_sources": probe.inventory(runtime), "python_resolved": str(Path(sys.executable).resolve())}
            probe.save(state / "protocol.json", protocol)
            (state / "protocol.sha256").write_text(probe.sha256(state / "protocol.json"))
            with patch.object(probe, "BASE", state):
                probe.frozen(state)
                for path in (source, inputs):
                    original = path.read_bytes()
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                        probe.frozen(state)
                    path.write_bytes(original)
                (runtime / "injected.py").write_text("new import")
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    probe.frozen(state)
                (state / "protocol.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                    probe.frozen(state)

    def test_failed_baseline_or_provenance_never_launches_candidate(self):
        for failure in ("digest", "provenance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                measured = baseline()
                measured["raw_logits"]["raw_sha256"] = "bad"
                with patch.object(probe, "frozen", side_effect=ValueError("source changed") if failure == "provenance" else None, return_value={}), patch.object(probe, "verify"), patch.object(probe, "health", return_value=200) as health, patch.object(probe, "power", return_value="power"), patch.object(probe.ReplayGuard, "check"), patch.object(probe.os, "getpriority", return_value=10), patch.object(probe, "run_arm", return_value=measured) as run:
                    with self.assertRaises(ValueError if failure == "digest" else RuntimeError):
                        probe.execute(state)
                self.assertEqual([call.args[1] for call in run.call_args_list], ["baseline"] if failure == "digest" else [])
                self.assertEqual(health.call_count, 2)
                result = json.loads((state / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertTrue((state / "cleanup.json").exists())
                self.assertFalse((state / "baseline-gate.json").exists())

    def test_successful_gate_precedes_candidate_and_refuses_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            arms = []

            def run(_, arm, *args):
                arms.append(arm)
                if arm == "candidate":
                    self.assertTrue(json.loads((state / "baseline-gate.json").read_text())["passed"])
                return baseline()

            with patch.object(probe, "frozen", return_value={}), patch.object(probe, "verify"), patch.object(probe, "health", return_value=200), patch.object(probe, "power", return_value="power"), patch.object(probe.ReplayGuard, "check"), patch.object(probe.os, "getpriority", return_value=10), patch.object(probe, "run_arm", side_effect=run):
                probe.execute(state)
                original = (state / "result.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    probe.execute(state)
                self.assertEqual((state / "result.json").read_bytes(), original)
            self.assertEqual(arms, ["baseline", "candidate"])

    def test_arm_deadline_counts_worker_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            guard = Mock(children=[], samples=[])
            child = guard.launch.return_value
            child.poll.return_value = None
            with patch.object(probe.time, "monotonic", side_effect=[0, 301]):
                with self.assertRaisesRegex(TimeoutError, "including setup"):
                    probe.run_arm(state, "baseline", {"commands": {"baseline": ["unused"]}}, guard)
            guard.launch.assert_called_once()

    def test_cancellation_reaps_worker_even_when_it_returns_zero(self):
        script = """
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def finish(*_):
    child.wait(timeout=5)
    sys.exit(0)
signal.signal(signal.SIGTERM, finish)
os.kill(os.getppid(), signal.SIGTERM)
while True: time.sleep(0.05)
"""
        handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            protocol = {"commands": {"baseline": [sys.executable, "-c", script]}}
            with patch.object(probe, "frozen", return_value=protocol), patch.object(probe, "health", return_value=200), patch.object(probe, "power", return_value="power"), patch("short_windows.power", return_value="power"), patch.object(probe.os, "getpriority", return_value=10):
                with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                    probe.execute(state)
            cleanup = json.loads((state / "cleanup.json").read_text())
            self.assertEqual(cleanup["errors"], [])
            child, = cleanup["children"]
            self.assertEqual(child["returncode"], 0)
            self.assertEqual(probe.group_members(child["pid"]), [])
            self.assertFalse((state / "candidate").exists())
            self.assertEqual(json.loads((state / "result.json").read_text())["status"], "failed")
        self.assertEqual({number: signal.getsignal(number) for number in handlers}, handlers)

    def test_deadline_still_attempts_cleanup_pins_and_production_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with patch.object(probe, "frozen", return_value={}) as frozen, patch.object(probe, "health", return_value=200) as health, patch.object(probe, "power", return_value="power"), patch.object(probe.os, "getpriority", return_value=10), patch.object(probe.ReplayGuard, "check"), patch.object(probe, "run_arm", side_effect=TimeoutError("overall work deadline")):
                with self.assertRaisesRegex(TimeoutError, "overall work deadline"):
                    probe.execute(state)
            self.assertEqual(frozen.call_count, 2)
            self.assertEqual(health.call_count, 2)
            self.assertEqual(json.loads((state / "cleanup.json").read_text())["errors"], [])
            result = json.loads((state / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["frozen_files"])
            self.assertEqual(result["production_after"], 200)

    def test_cleanup_escalates_uncooperative_owned_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            guard = probe.ReplayGuard(state)
            child = guard.launch([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"], state / "worker.log")
            try:
                import time
                until = time.monotonic() + 5
                while "ready" not in (state / "worker.log").read_text():
                    if time.monotonic() >= until:
                        self.fail("CPU worker did not start")
                    time.sleep(0.05)
                guard.close()
                self.assertEqual(child.returncode, -signal.SIGKILL)
                self.assertEqual(probe.group_members(child.pid), [])
                self.assertEqual(json.loads((state / "cleanup.json").read_text())["errors"], [])
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
