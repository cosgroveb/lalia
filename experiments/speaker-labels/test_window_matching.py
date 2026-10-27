from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import window_matching
from window_matching import clean_spans, nearest, normalize, ready, select_clips


class WindowMatchingTests(unittest.TestCase):
    def test_overlap_is_removed_but_same_speaker_intervals_merge(self):
        spans = clean_spans([{"start": 0, "end": 10, "speaker_id": "a"},
                             {"start": 2, "end": 4, "speaker_id": "a"},
                             {"start": 5, "end": 6, "speaker_id": "b"}])
        self.assertEqual(spans, [{"start": 0, "end": 5, "speaker_id": "a"},
                                 {"start": 6, "end": 10, "speaker_id": "a"}])

    def fixture(self):
        turns = [{"start": region * 100 + speaker * 10,
                  "end": region * 100 + speaker * 10 + 3.5, "speaker_id": str(speaker)}
                 for region in range(6) for speaker in range(4)]
        return {"reference_turns": turns, "reference_segments": deepcopy(turns)}

    def test_selection_is_disjoint_deterministic_and_spans_regions(self):
        fixture = self.fixture()
        samples, counts = select_clips(fixture)
        self.assertEqual(select_clips(fixture), (samples, counts))
        self.assertEqual(len(samples), 24)
        ordered = sorted(samples, key=lambda sample: sample["start_frame"])
        self.assertTrue(all(a["end_frame"] <= b["start_frame"] for a, b in zip(ordered, ordered[1:])))
        self.assertTrue(all(sample["end_frame"] - sample["start_frame"] == 48000 for sample in samples))
        self.assertTrue(all(count["minimum_separation_seconds"] >= 30 for count in counts.values()))
        self.assertEqual(fixture, self.fixture())

    def test_shortage_and_manual_cross_speaker_contamination_fail(self):
        fixture = self.fixture()
        fixture["reference_segments"].append({"start": 200, "end": 204, "speaker_id": "1"})
        with self.assertRaisesRegex(ValueError, "Insufficient later held-out"):
            select_clips(fixture)
        fixture = self.fixture()
        fixture["reference_turns"] = fixture["reference_turns"][:-4]
        with self.assertRaisesRegex(ValueError, "Insufficient later held-out"):
            select_clips(fixture)

    def test_two_second_crops_preserve_original_selection_and_update_counts(self):
        fixture = self.fixture()
        short_span = {"start": 55, "end": 57.5, "speaker_id": "0"}
        fixture["reference_turns"].append(short_span)
        fixture["reference_segments"].append(short_span)
        original, original_counts = select_clips(fixture)
        self.assertEqual((original, original_counts), select_clips(fixture, 3))
        shortened, counts = select_clips(fixture, 2)
        self.assertEqual(shortened, [dict(sample, end_frame=sample["start_frame"] + 32000)
                                     for sample in original])
        self.assertFalse(any(sample["start_frame"] == int(55.15 * 16000) for sample in shortened))
        for speaker, values in original_counts.items():
            self.assertEqual(counts[speaker], dict(values, enrollment_seconds=4, evaluation_seconds=8,
                                                  minimum_separation_seconds=
                                                  values["minimum_separation_seconds"] + 1))
        with self.assertRaisesRegex(ValueError, "two or three seconds"):
            select_clips(fixture, 1)

    def test_embed_uses_requested_duration_and_keeps_three_second_default(self):
        session = Mock()
        session.get_inputs.return_value = [Mock(name="input")]
        session.get_inputs.return_value[0].name = "features"
        session.run.return_value = [np.ones((1, 192), dtype=np.float32)]
        two_seconds = np.ones(32000, dtype="<i2").tobytes()
        with self.assertRaisesRegex(ValueError, "Expected 3 seconds"):
            window_matching.embed(two_seconds, session)
        session.run.assert_not_called()
        vector = window_matching.embed(two_seconds, session, 2)
        self.assertEqual(session.run.call_args.args[1]["features"].shape, (1, 198, 80))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
        window_matching.embed(np.ones(48000, dtype="<i2").tobytes(), session)
        self.assertEqual(session.run.call_args.args[1]["features"].shape, (1, 298, 80))

    def test_frozen_matcher_rejects_and_does_not_use_evaluation_labels(self):
        centroids = {"a": normalize([1, 0, 0]), "b": normalize([0, 1, 0])}
        before = {key: value.copy() for key, value in centroids.items()}
        self.assertEqual(nearest(normalize([1, 0, 0]), centroids)[:2], ("a", "a"))
        self.assertEqual(nearest(normalize([0.5, 0.4, 1]), centroids)[:2], ("a", None))
        for speaker in centroids:
            np.testing.assert_array_equal(centroids[speaker], before[speaker])
        with self.assertRaisesRegex(ValueError, "Invalid embedding"):
            normalize([float("nan"), 0])

    def test_readiness_requires_four_speakers_and_no_rejects_or_errors(self):
        counts = {speaker: {"correct_accepted": 4, "wrong_accepted": 0, "rejected": 0}
                  for speaker in ("a", "b", "c", "d")}
        self.assertTrue(ready(counts))
        self.assertFalse(ready({}))
        for field in ("wrong_accepted", "rejected"):
            altered = deepcopy(counts)
            altered["a"][field] = 1
            self.assertFalse(ready(altered))
        for values in counts.values():
            values.update(correct_accepted=0, rejected=4)
        self.assertFalse(ready(counts))

    def test_state_write_failure_and_sigterm_reap_owned_worker(self):
        spawn, save = subprocess.Popen, window_matching.save
        handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
        for failure in ("state-write", "SIGTERM"):
            processes = []

            def start_worker(*args, **kwargs):
                process = spawn([sys.executable, "-c", "import time; time.sleep(30)"],
                                start_new_session=True)
                processes.append(process)
                return process

            def save_state(path, value):
                if path.name == "process.json" and value["state"] == "running":
                    if failure == "state-write":
                        raise OSError("Simulated state-write failure")
                    os.kill(os.getpid(), signal.SIGTERM)
                save(path, value)

            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, model = root / "manifest.json", root / "model.onnx"
                model.write_bytes(b"unused model")
                clip = dict(self.fixture(), id="ami-ES2004a-full", private=False, audio_path=str(model))
                manifest.write_text(json.dumps({"clips": [clip]}))
                argv = ["window_matching.py", "--manifest", str(manifest), "--model", str(model),
                        "--output", str(root)]
                try:
                    with patch.object(sys, "argv", argv), \
                            patch("window_matching.sha256", return_value=window_matching.MODEL_SHA256), \
                            patch("window_matching.battery", return_value=100), \
                            patch("window_matching.subprocess.Popen", side_effect=start_worker), \
                            patch("window_matching.save", side_effect=save_state):
                        with self.assertRaises(OSError if failure == "state-write" else KeyboardInterrupt):
                            window_matching.main()
                    self.assertEqual(len(processes), 1)
                    self.assertIsNotNone(processes[0].returncode)
                    self.assertEqual(json.loads((root / "process.json").read_text())["state"], "exited")
                finally:
                    for process in processes:
                        window_matching.stop(process)
                    for number, handler in handlers.items():
                        signal.signal(number, handler)

    def test_signal_races_reap_exited_child_and_continue_remaining_cleanup(self):
        for race_signal in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(race_signal=race_signal):
                exiting, remaining = Mock(pid=101), Mock(pid=102)
                exiting.poll.return_value = remaining.poll.return_value = None
                if race_signal == signal.SIGKILL:
                    exiting.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]

                def signal_group(pid, number):
                    if pid == exiting.pid and number == race_signal:
                        raise ProcessLookupError("Child exited before signal")

                with patch("window_matching.os.killpg", side_effect=signal_group) as killpg:
                    for process in (exiting, remaining):
                        window_matching.stop(process)

                for process in (exiting, remaining):
                    self.assertEqual(process.wait.call_args, call())
                killpg.assert_any_call(exiting.pid, race_signal)
                killpg.assert_any_call(remaining.pid, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
