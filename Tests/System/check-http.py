"""Exercise the signed app's real HTTP client without inference or capture."""
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave


OFFICIAL_PROMPT = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.paths.append(self.path)
        assert self.path == "/v1/models", self.path
        self.reply({"data": [{"id": "cohere-transcribe-03-2026"}, {"id": "fixture-session"}]})

    def do_POST(self):
        self.server.paths.append(self.path)
        assert self.path == "/v1/audio/transcriptions", self.path
        body = self.rfile.read(int(self.headers["Content-Length"]))
        headers = f"Content-Type: {self.headers['Content-Type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        parts = BytesParser(policy=default).parsebytes(headers + body)
        fields = {}
        for part in parts.iter_parts():
            fields[part.get_param("name", header="content-disposition")] = part.get_payload(decode=True)
        session = fields["model"] == b"fixture-session"
        assert fields["model"] in (b"cohere-transcribe-03-2026", b"fixture-session")
        assert fields["language"] == b"en"
        assert fields["max_tokens"] == (b"49152" if session else b"1024")
        assert fields["stream"] == b"false"
        assert fields.get("prompt") == (OFFICIAL_PROMPT.encode() if session else None)
        assert len(fields["file"]) <= (250_000_000 if session else 20_000_000)
        with wave.open(io.BytesIO(fields["file"]), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            frames = audio.getnframes()
        assert frames <= (5400 if session else 600) * 16000
        self.server.frames.append(frames)
        if self.server.redirect:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/must-not-follow")
            self.end_headers()
        elif session:
            result = {
                "text": f"[0][S01]नमस्ते 🫠[1][1][S02]fixture session[{frames / 16000}]",
                "duration": 99999,
                "segments": [
                    {"start": 0, "end": 1, "text": "[S01] नमस्ते 🫠", "speaker_id": "S01"},
                    {"start": 1, "end": frames / 16000, "text": "[S02] fixture session", "speaker_id": "S02"},
                ],
                "usage": {"type": "tokens", "input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            }
            if self.server.invalid_usage:
                del result["usage"]
            self.reply(result)
        else:
            self.reply({"text": "fixture transcript", "duration": 99999})

    def reply(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    app = str(Path(sys.argv[1]).resolve())
    with HTTPServer(("127.0.0.1", 0), Handler) as server, ThreadPoolExecutor(max_workers=1) as executor:
        server.frames = []
        server.paths = []
        server.redirect = False
        server.invalid_usage = False
        worker = executor.submit(server.serve_forever)
        try:
            with tempfile.TemporaryDirectory(prefix="lalia-http-check-") as directory:
                root = Path(directory)
                fixture = root / "fixture.wav"
                with wave.open(str(fixture), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    second = struct.pack("<h", 1000) * 16000
                    for _ in range(610):
                        audio.writeframesraw(second)
                config = root / "config.json"
                config.write_text(json.dumps({
                    "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                    "transcriptionModel": "cohere-transcribe-03-2026",
                    "sessionTranscriptionModel": "fixture-session",
                    "language": "en", "summaryModel": "fixture-notes",
                }))
                command = [app, "--transcribe-file", str(fixture), "--config", str(config)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                assert result.returncode == 0, result.stderr
                assert len(server.frames) == 2, server.frames
                assert sum(server.frames) == 610 * 16000, server.frames
                assert result.stdout.count("fixture transcript") == 2, result.stdout
                assert "Microphone (approximate)" in result.stdout
                assert "00:10:10" in result.stdout, result.stdout
                server.redirect = True
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                assert result.returncode != 0, result.stdout
                assert "HTTP 302" in result.stderr, result.stderr
                assert "/must-not-follow" not in server.paths, server.paths
                assert fixture.exists()
                server.redirect = False
                server.frames = []
                with wave.open(str(fixture), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    for _ in range(900):
                        audio.writeframesraw(second)
                command[1] = "--transcribe-session-file"
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                assert result.returncode == 0, result.stderr
                assert server.frames == [900 * 16000], server.frames
                assert "Speaker 1\nनमस्ते 🫠" in result.stdout, result.stdout
                assert "Speaker 2\nfixture session" in result.stdout, result.stdout
                assert "00:15:00" in result.stdout, result.stdout
                server.invalid_usage = True
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                assert result.returncode != 0, result.stdout
                assert "completion could not be verified" in result.stderr, result.stderr
                assert fixture.exists()
                print("PASS: signed app dictation/session routing, Unicode speakers, bounded multipart, usage rejection, frame conservation, and redirects (fixtures only)")
        finally:
            server.shutdown()
            worker.result()


if __name__ == "__main__":
    main()
