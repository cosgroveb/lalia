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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.paths.append(self.path)
        assert self.path == "/v1/models", self.path
        self.reply({"data": [{"id": "cohere-transcribe-03-2026"}]})

    def do_POST(self):
        self.server.paths.append(self.path)
        assert self.path == "/v1/audio/transcriptions", self.path
        body = self.rfile.read(int(self.headers["Content-Length"]))
        headers = f"Content-Type: {self.headers['Content-Type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        parts = BytesParser(policy=default).parsebytes(headers + body)
        fields = {}
        for part in parts.iter_parts():
            fields[part.get_param("name", header="content-disposition")] = part.get_payload(decode=True)
        assert fields["model"] == b"cohere-transcribe-03-2026"
        assert fields["language"] == b"en"
        assert fields["max_tokens"] == b"1024"
        assert fields["stream"] == b"false"
        assert len(fields["file"]) <= 20_000_000
        with wave.open(io.BytesIO(fields["file"]), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            frames = audio.getnframes()
        assert frames <= 600 * 16000
        self.server.frames.append(frames)
        if self.server.redirect:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/must-not-follow")
            self.end_headers()
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
                print("PASS: signed app HTTP, bounded multipart, frame conservation, and redirect rejection (fixture responses only)")
        finally:
            server.shutdown()
            worker.result()


if __name__ == "__main__":
    main()
