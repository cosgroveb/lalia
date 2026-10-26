import contextlib
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run


class RunnerTests(unittest.TestCase):
    def test_curl_sends_prompt_as_literal_multipart_text(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                headers = f"Content-Type: {self.headers['Content-Type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                parts = BytesParser(policy=default).parsebytes(headers + self.rfile.read(int(self.headers["Content-Length"])))
                self.server.fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                                      for part in parts.iter_parts()}
                data = b'{"text":""}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        with tempfile.TemporaryDirectory() as temporary, HTTPServer(("127.0.0.1", 0), Handler) as http, \
                ThreadPoolExecutor(max_workers=1) as executor:
            directory = Path(temporary)
            audio = directory / "audio.wav"
            audio.write_bytes(b"audio fixture")
            prompt = "@/not/a/file;type=text/plain\r\n请转写音频 🫠\n"
            args = SimpleNamespace(max_tokens=49152, timeout=10, deadline=None, port=http.server_port,
                                   prompt=prompt, memory_gib=64, min_battery_percent=25, qualify=False)
            server = SimpleNamespace(poll=lambda: None, pid=0)
            worker = executor.submit(http.serve_forever)
            try:
                with patch.object(run, "footprint", return_value=1024), \
                        patch.object(run.subprocess, "run", return_value=SimpleNamespace(stdout="AC Power")), \
                        patch.object(run, "score", return_value={}):
                    result = run.transcribe(args, server, "MOSS-Transcribe-Diarize",
                                            {"audio_path": str(audio), "duration": 10}, directory)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(http.fields["prompt"], prompt.encode())
                self.assertEqual(http.fields["max_tokens"], b"49152")
                self.assertEqual(http.fields["file"], audio.read_bytes())
            finally:
                http.shutdown()
                worker.result()

    def test_prompt_file_is_frozen_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest, prompt = state / "manifest.json", state / "prompt.txt"
            manifest.write_text(json.dumps([{"id": "fixture", "duration": 10}]))
            original = "请转写音频 🫠\r\n"
            prompt.write_bytes(original.encode())
            argv = ["run.py", "--state", str(state), "--manifest", str(manifest), "--prompt-file", str(prompt),
                    "--models", "MOSS-Transcribe-Diarize"]

            def transcribe(args, *_):
                prompt.write_text("changed after loading")
                self.assertEqual(args.prompt, original)
                return {"status": "complete"}

            with patch.object(sys, "argv", argv), patch.object(run.signal, "signal"), \
                    patch.object(run.subprocess, "Popen"), patch.object(run, "start_server"), \
                    patch.object(run, "stop"), patch.object(run, "transcribe", side_effect=transcribe), \
                    patch.object(run.subprocess, "run", return_value=SimpleNamespace(stdout="swap")), \
                    contextlib.redirect_stdout(io.StringIO()):
                # Process records need numeric PIDs, as the real subprocess objects provide.
                run.subprocess.Popen.return_value.pid = 1
                run.main()
                with self.assertRaisesRegex(ValueError, "different or unrecorded prompt"):
                    run.main()
                run.start_server.assert_called_once()
            result = json.loads((state / "results/MOSS-Transcribe-Diarize/fixture/result.json").read_text())
            self.assertEqual(result["prompt"], original)
            self.assertEqual(result["prompt_sha256"], hashlib.sha256(original.encode()).hexdigest())
            self.assertEqual(Path(result["prompt_source"]), prompt.resolve())

    def test_changed_or_missing_prompt_baseline_does_not_start_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest, baseline, prompt = state / "manifest.json", state / "baseline.json", state / "prompt.txt"
            manifest.write_text(json.dumps([{"id": "fixture", "duration": 3600}]))
            prompt.write_text("请转写音频 🫠")
            document = {"status": "complete", "model": "MOSS-Transcribe-Diarize", "audio_seconds": 1049,
                        "completion": {"eos": True, "raw_coverage": True},
                        "qualification": {"accepted": True, "failures": []},
                        "metrics": {"schema": {"speaker_count": 4, "missing_speaker_count": 0},
                                    "speaker_attributed": {"cpwer": 0.1, "missed_speakers": 0, "extra_speakers": 0},
                                    "diarization": {"boundary_250ms": {"der": 0.1}}}}
            argv = ["run.py", "--state", str(state), "--manifest", str(manifest), "--qualify",
                    "--baseline", str(baseline), "--prompt-file", str(prompt)]
            for prior in ({}, {"prompt": "different"}, {"prompt": prompt.read_text(), "prompt_sha256": "wrong"}):
                baseline.write_text(json.dumps({**document, **prior}))
                with self.subTest(prior=prior), patch.object(sys, "argv", argv), \
                        patch.object(run.subprocess, "Popen") as popen, patch.object(run.signal, "signal"):
                    with self.assertRaisesRegex(ValueError, "same prompt"):
                        run.main()
                    popen.assert_not_called()

    def test_invalid_baseline_does_not_start_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = state / "manifest.json"
            baseline = state / "baseline.json"
            argv = ["run.py", "--state", str(state), "--manifest", str(manifest), "--qualify"]
            for duration in (1049, 3600):
                manifest.write_text(json.dumps([{"id": "fixture", "duration": duration}]))
                for document in (None, "{", "null", "[]", "{}", '{"status":"failed"}'):
                    if document is None and duration < 3600:
                        continue
                    if document is not None:
                        baseline.write_text(document)
                    with self.subTest(document=document, duration=duration), \
                            patch.object(sys, "argv", argv + (["--baseline", str(baseline)] if document is not None else [])), \
                            patch.object(run.subprocess, "Popen") as popen, patch.object(run.signal, "signal"):
                        with self.assertRaises(ValueError):
                            run.main()
                        popen.assert_not_called()

    def test_battery_cutoff_stops_batch_and_waits_for_children(self):
        children = []
        popen = subprocess.Popen

        def child(*_, **__):
            process = popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            children.append(process)
            return process

        def command(args, **_):
            return SimpleNamespace(stdout="Now drawing from 'Battery Power'\n25%; discharging" if args[0] == "pmset" else "swap")

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            manifest = state / "manifest.json"
            manifest.write_text(json.dumps([{"id": name, "duration": 10, "audio_path": "unused.wav"}
                                            for name in ("first", "second")]))
            argv = ["run.py", "--state", str(state), "--manifest", str(manifest), "--models", "MOSS-Transcribe-Diarize"]
            try:
                with patch.object(sys, "argv", argv), patch.object(run.subprocess, "Popen", side_effect=child), \
                        patch.object(run.subprocess, "run", side_effect=command), \
                        patch.object(run, "start_server", side_effect=child) as start_server, \
                        patch.object(run, "footprint", return_value=1024), \
                        patch.object(run.signal, "signal"), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(run.BatteryLimitError, "Battery reached 25%"):
                        run.main()
                self.assertEqual(start_server.call_count, 1)
                self.assertEqual(len(children), 3)
                self.assertFalse((state / "results/MOSS-Transcribe-Diarize/second").exists())
                self.assertEqual(json.loads((state / "progress.json").read_text())["status"], "interrupted")
                self.assertTrue(all(process.returncode == -signal.SIGTERM for process in children))
            finally:
                for process in children:
                    if process.poll() is None:
                        run.stop(process)


if __name__ == "__main__":
    unittest.main()
