"""Opt-in real MOSS containment replay against an isolated runtime.

Use the runtime's Python (NumPy required):
  python check-runtime-limits.py --url http://127.0.0.1:8003 --pid PID --speech speech.wav
An optional --audio WAV replays a saved failure instead of synthetic noise.
All requests cap output at 512 tokens, including on an unpatched runtime.
The supplied speech must contain words, as 16 kHz mono PCM16 WAV.
"""

import argparse
import http.client
import json
import re
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
import wave

import numpy as np

PROMPT = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"


def write_audio(path, samples):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes((samples * 32767).astype("<i2").tobytes())


def cpu_seconds(pid):
    value = subprocess.check_output(["ps", "-p", str(pid), "-o", "time="], text=True).strip()
    fields = value.split(":")
    assert 2 <= len(fields) <= 3, value
    return sum(float(field) * 60 ** index for index, field in enumerate(reversed(fields)))


def request(url, path, timeout=30, disconnect=False):
    boundary = "LaliaRuntimeLimitTest"
    parts = []
    for name, value in {"model": "MOSS-Transcribe-Diarize", "language": "en", "prompt": PROMPT,
                        "max_tokens": "512", "timeout_seconds": str(timeout)}.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode())
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=15)
    started = time.monotonic()
    try:
        connection.request("POST", "/v1/audio/transcriptions", b"".join(parts),
                           {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        if disconnect:
            time.sleep(0.2)
            return None, None, time.monotonic() - started
        response = connection.getresponse()
        return response.status, json.loads(response.read()), time.monotonic() - started
    finally:
        connection.close()


def spoken_text(result, duration):
    segments = result.get("segments")
    assert segments, result
    for segment in segments:
        assert 0 <= segment["start"] <= segment["end"] <= duration + 0.1, segment
    text = " ".join(re.sub(r"\[[^\]]+\]", "", result.get("text", "")).split())
    assert text, result
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    url = urlsplit(args.url)
    assert url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost"), "Use an isolated loopback runtime"
    with tempfile.TemporaryDirectory(prefix="lalia-limits-") as directory:
        root = Path(directory)
        noise = np.random.default_rng(1).normal(size=29740)
        noise *= 10 ** (-49.7 / 20) / np.sqrt(np.mean(noise * noise))
        write_audio(root / "noise.wav", noise)
        bad = args.audio or root / "noise.wav"
        with wave.open(str(args.speech), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            speech = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float) / 32767
        rms = np.sqrt(np.mean(speech * speech))
        assert rms > 0, "Speech fixture cannot be silent"
        write_audio(root / "quiet.wav", speech * 10 ** (-49.7 / 20) / rms)
        status, valid, _ = request(url, args.speech)
        assert status == 200, (status, valid)
        expected_words = spoken_text(valid, len(speech) / 16000)
        report = []
        for name, kwargs in [("budget", {}), ("deadline", {"timeout": 0.2}), ("disconnect", {"disconnect": True})]:
            status, body, elapsed = request(url, bad, **kwargs)
            if name != "disconnect":
                expected = "token budget" if name == "budget" else "time limit"
                assert status == (500 if name == "budget" else 504) and expected in json.dumps(body), (status, body)
                if name == "deadline":
                    assert elapsed < 1, elapsed
            time.sleep(0.3)  # Allow the in-flight GPU token to drain.
            before = cpu_seconds(args.pid)
            time.sleep(0.5)
            cpu_delta = cpu_seconds(args.pid) - before
            assert cpu_delta <= 0.03, f"{name}: worker still computing ({cpu_delta:.2f}s CPU)"
            next_status, next_body, next_elapsed = request(url, args.speech)
            assert next_status == 200 and spoken_text(next_body, len(speech) / 16000) == expected_words, (next_status, next_body)
            report.append(dict(case=name, elapsed=round(elapsed, 3), idle_cpu_delta=round(cpu_delta, 3), next_elapsed=round(next_elapsed, 3)))
        status, quiet, _ = request(url, root / "quiet.wav")
        assert status == 200 and spoken_text(quiet, len(speech) / 16000) == expected_words, (status, quiet)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
