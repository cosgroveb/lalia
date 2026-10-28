import copy
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import limit60
from limit60_runtime import DENOMINATOR_SHA, PROMPT_SHA, check_arguments
from limit60_score import compare_tail, native_finish, quarter_continuity, tail_metrics


def segment(start, end, speaker, text):
    return {"start": start, "end": end, "speaker_id": speaker, "text": text}


class NativeAuditTests(unittest.TestCase):
    def evidence(self):
        rows = [{"loaded_layers": 28, "denominator_sha256": DENOMINATOR_SHA, "weights_dtype": "bfloat16"},
                {"model_max_tokens": 49152, "prompt_sha256": PROMPT_SHA, "eos_ids": [151645]},
                {"builder_prompt_sha256": PROMPT_SHA, "input_tokens": 100},
                {"generator_max_tokens": 49152, "generator_input_tokens": 100},
                {"yielded_tokens": 201, "last_token_id": 151645}]
        response = {"usage": {"type": "tokens", "input_tokens": 100, "output_tokens": 200, "total_tokens": 300}}
        return rows, response

    def test_native_eos_requires_consistent_constructor_and_usage(self):
        rows, response = self.evidence()
        self.assertEqual(native_finish(rows, response)["yielded_tokens"], 201)
        for index, key, value in ((0, "loaded_layers", 27), (0, "weights_dtype", "float32"),
                                  (2, "input_tokens", 101), (4, "last_token_id", 42),
                                  (4, "yielded_tokens", 200)):
            changed = copy.deepcopy(rows)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_finish(changed, response)

    def test_missing_or_duplicate_terminal_audit_is_not_success(self):
        rows, response = self.evidence()
        for changed in (rows[:-1], rows + [rows[-1]]):
            with self.assertRaisesRegex(ValueError, "Expected one"):
                native_finish(changed, response)

    def test_count_below_ceiling_cannot_substitute_for_native_eos(self):
        rows, response = self.evidence()
        rows[-1]["last_token_id"] = 151643
        with self.assertRaises(ValueError):
            native_finish(rows, response)

    def test_launcher_rejects_production_or_other_experiment_paths(self):
        state = Path("/tmp/limit60-test")
        base = state / "a-control/server/MOSS-Transcribe-Diarize"
        args = ["launcher", "serve", "--host", "127.0.0.1", "--port", "18001",
                "--base-path", str(base), "--model-dir", str(base / "models")]
        check_arguments(args, state)
        for flag, value in (("--port", "8000"), ("--host", "0.0.0.0"),
                            ("--base-path", "/tmp/other"), ("--model-dir", "/tmp/other")):
            changed = list(args)
            changed[changed.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                check_arguments(changed, state)


class TailTests(unittest.TestCase):
    def test_tail_crop_preserves_boundary_text_and_reports_it(self):
        reference = [segment(99, 101, "A", "boundary"), segment(110, 112, "B", "hello")]
        clip = {"id": "tail", "duration": 400, "reference_segments": reference, "reference_turns": reference}
        result = tail_metrics(clip, {"segments": reference})
        self.assertEqual(result["boundary_hypothesis_segments"], 1)
        self.assertEqual(result["metrics"]["speaker_attributed"]["cpwer"], 0)
        self.assertEqual(result["metrics"]["schema"]["reference_last_word_end"], 12)

    def metrics(self, cpwer=0.15, der=0.1, coverage=0.95):
        return {"speaker_attributed": {"cpwer": cpwer},
                "diarization": {"boundary_250ms": {"der": der},
                                "speech_coverage": {"covered_reference_fraction": coverage}}}

    def test_tail_regression_boundary_and_missing_coverage(self):
        baseline = self.metrics()
        self.assertFalse(compare_tail(self.metrics(cpwer=0.2, der=0.15, coverage=0.9), baseline)["failures"])
        self.assertEqual(len(compare_tail(self.metrics(cpwer=0.201, der=0.151, coverage=0.899), baseline)["failures"]), 3)
        for missing in (None, float("nan")):
            with self.assertRaises(ValueError):
                compare_tail(self.metrics(coverage=missing), baseline)

    def test_last_quarter_identity_switch_cannot_hide_in_global_mapping(self):
        refs, hyps = [], []
        for quarter in range(4):
            for offset, speaker, text in ((10, "A", "alpha one two three"), (20, "B", "bravo four five six")):
                item = segment(900 * quarter + offset, 900 * quarter + offset + 4, speaker, text)
                refs.append(item)
                label = {"A": "S01", "B": "S02"}[speaker]
                if quarter == 3:
                    label = {"S01": "S02", "S02": "S01"}[label]
                hyps.append(dict(item, speaker_id=label))
        result = quarter_continuity({"reference_segments": refs}, {"segments": hyps})
        self.assertEqual([q["global_assignment_agrees"] for q in result["quarters"]], [True, True, True, False])


class PhaseTests(unittest.TestCase):
    def test_interruption_reaches_runner_and_waits_for_child_cleanup(self):
        spawn, command = subprocess.Popen, subprocess.run
        runner = """
import os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from run import interrupt, stop
state = Path(sys.argv[1])
def received(number, frame):
    (state / "received").write_text(str(number))
    interrupt(number, frame)
signal.signal(signal.SIGTERM, received)
signal.signal(signal.SIGINT, received)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
try:
    (state / "ready").write_text(str(child.pid))
    if int(sys.argv[2]) == signal.SIGINT:
        time.sleep(0.1)
        os.kill(os.getppid(), signal.SIGINT)
    time.sleep(2)
except KeyboardInterrupt:
    pass
finally:
    os.kill(os.getppid(), int(sys.argv[2]))
    time.sleep(0.1)
    stop(child)
    (state / "reaped").write_text(str(child.returncode))
"""
        handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
        for number, handler in handlers.items():
            self.addCleanup(signal.signal, number, handler)
            signal.signal(number, lambda *_: None)
        for number in handlers:
            with self.subTest(signal=number), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary)
                phase = state / "a-control"
                phase.mkdir()
                protocol = {"hashes": {}, "commands": {"a-control": [sys.executable, "-c", runner,
                            str(state), str(number), str(limit60.SOURCE)]}}
                (state / "protocol.json").write_text(json.dumps(protocol))
                (state / "protocol.sha256").write_text(limit60.sha256(state / "protocol.json"))
                processes = []

                def start(*args, **kwargs):
                    process = spawn(*args, **kwargs)
                    processes.append(process)
                    deadline = time.monotonic() + 5
                    while not (state / "ready").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("Synthetic runner did not become ready")
                        time.sleep(0.01)
                    if number == signal.SIGTERM:
                        os.kill(os.getpid(), number)
                    return process

                def run(args, **kwargs):
                    return SimpleNamespace(stdout="fixture power") if args[0] == "pmset" else command(args, **kwargs)

                previous = {number: signal.getsignal(number) for number in handlers}
                try:
                    with patch.object(limit60.subprocess, "Popen", side_effect=start), \
                            patch.object(limit60.subprocess, "run", side_effect=run), \
                            patch.object(limit60.urllib.request, "urlopen") as urlopen, \
                            patch.object(limit60, "cleanup_evidence", return_value={"complete": True}):
                        urlopen.return_value.__enter__.return_value.status = 200
                        result = limit60.run_phase(state, "a-control")
                    self.assertTrue((state / "received").exists(), "Launcher did not forward interruption")
                    self.assertEqual(int((state / "received").read_text()), number)
                    self.assertEqual(int((state / "reaped").read_text()), -signal.SIGTERM)
                    self.assertEqual(processes[0].returncode, 0)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int((state / "ready").read_text()), 0)
                    self.assertFalse(result["accepted"])
                    self.assertIn(f"Runner interrupted by {signal.Signals(number).name}", result["failures"])
                    self.assertEqual({number: signal.getsignal(number) for number in handlers}, previous)
                finally:
                    for process in processes:
                        process.terminate()
                        process.wait(timeout=5)

    def test_crop_duration_matches_written_samples_at_large_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.wav", Path(directory) / "crop.wav"
            data_bytes = 59085910 * 2
            header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", data_bytes + 36, b"WAVE",
                                 b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data", data_bytes)
            with source.open("wb") as output:
                output.write(header)
                output.truncate(44 + data_bytes)
            start, end = 42296235, 42296251
            duration = limit60.crop_wave(source, target, start, end)
            with wave.open(str(target)) as audio:
                self.assertEqual(duration, audio.getnframes() / audio.getframerate())
            self.assertEqual(duration, 0.001)
            self.assertNotEqual(duration, end / 16000 - start / 16000)

    def protocol(self, state):
        (state / "protocol.json").write_text(json.dumps({"hashes": {}}))
        (state / "protocol.sha256").write_text(limit60.sha256(state / "protocol.json"))

    def test_rejected_prior_case_prevents_any_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            self.protocol(state)
            (state / "a-control").mkdir()
            (state / "a-control/evaluation.json").write_text('{"accepted": false}')
            with patch.object(limit60.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "Previous phase"):
                limit60.run_phase(state, "a-hour")
            run.assert_not_called()

    def test_started_phase_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            self.protocol(state)
            (state / "a-control").mkdir()
            (state / "a-control/started.json").write_text("{}")
            with patch.object(limit60.subprocess, "run") as run, self.assertRaises(FileExistsError):
                limit60.run_phase(state, "a-control")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
