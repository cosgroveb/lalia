#!/usr/bin/env python3
"""Private MOSS transcription runtime for Lalia."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave


BASE_URL = "http://127.0.0.1:8002/v1"
MODEL = "MOSS-Transcribe-Diarize"
REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
ROOT = Path.home() / ".local/share/lalia/runtime"
CONFIG = Path.home() / ".config/lalia/config.json"
WEIGHT_SIZE = 1817113576
WEIGHT_SHA256 = "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"
LEGACY_MODEL = "cohere-transcribe-03-2026"
OFFICIAL_PROMPT = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
REQUIRED_FILES = (
    "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "chat_template.jinja", "preprocessor_config.json",
    "processor_config.json", "generation_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "model-00000-of-00001.safetensors",
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


class PortConflictError(RuntimeError):
    pass


def request(path, payload=None, timeout=5):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def model_is_valid(path):
    try:
        weight = path / "model-00000-of-00001.safetensors"
        if (not all((path / name).is_file() for name in REQUIRED_FILES)
                or (path / ".source-revision").read_text().strip() != REVISION
                or weight.stat().st_size != WEIGHT_SIZE
                or json.loads((path / "config.json").read_text()).get("model_type")
                != "moss_transcribe_diarize"):
            return False
        digest = hashlib.sha256()
        with weight.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == WEIGHT_SHA256
    except (OSError, ValueError):
        return False


def provision_model():
    models = ROOT / "models"
    target = models / MODEL
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not model_is_valid(target):
            raise RuntimeError(f"Refusing incomplete or unrecognized model at {target}")
        return target

    legacy = Path.home() / ".omlx/models" / MODEL
    if model_is_valid(legacy):
        models.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=f".{MODEL}-copy-", dir=models))
        try:
            for name in REQUIRED_FILES:
                shutil.copy2(legacy / name, staged / name)
            shutil.copy2(legacy / ".source-revision", staged / ".source-revision")
            if not model_is_valid(staged):
                raise RuntimeError("Existing shared model changed or failed verification during copy")
            staged.rename(target)
        except BaseException:
            shutil.rmtree(staged)
            raise
    else:
        from huggingface_hub import snapshot_download
        staged = ROOT / "downloads" / f"{MODEL}-{REVISION}"
        snapshot_download(repo_id=REPO, revision=REVISION, local_dir=staged,
                          allow_patterns=["*.json", "*.model", "*.safetensors", "*.jinja", "*.txt"])
        (staged / ".source-revision").write_text(REVISION + "\n")
        if not model_is_valid(staged):
            raise RuntimeError(f"Incomplete model download retained at {staged}")
        models.mkdir(parents=True, exist_ok=True)
        staged.rename(target)
    return target


def write_config():
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG.read_bytes() if CONFIG.exists() else None
    config = json.loads(original or b"{}")
    current = config.get("baseURL", "")
    legacy_model = config.get("sessionTranscriptionModel")
    transcription_model = legacy_model if legacy_model is not None else config.get("transcriptionModel")
    migratable_model = transcription_model in (MODEL, LEGACY_MODEL)
    if current == "http://127.0.0.1:8000/v1" and not migratable_model:
        print("Preserved Lalia config because its transcription model is unknown or custom")
        return
    if current in ("", BASE_URL) and transcription_model not in (None, MODEL):
        print("Preserved Lalia config because its transcription model is custom")
        return
    if current and current not in ("http://127.0.0.1:8000/v1", BASE_URL):
        print(f"Preserved custom transcription endpoint: {current}")
        return
    if config.get("summaryBaseURL") not in (None, "", "http://127.0.0.1:8000/v1", BASE_URL):
        print(f"Preserved custom summary endpoint: {config['summaryBaseURL']}")
    updated = dict(config)
    updated["baseURL"] = BASE_URL
    if legacy_model is not None:
        updated["transcriptionModel"] = MODEL if transcription_model == LEGACY_MODEL else transcription_model
        updated.pop("sessionTranscriptionModel", None)
    elif transcription_model == LEGACY_MODEL:
        updated["transcriptionModel"] = MODEL
    else:
        updated.setdefault("transcriptionModel", MODEL)
    updated.setdefault("language", "en")
    updated.setdefault("summaryModel", "Qwen3.6-35B-A3B-oQ8-mtp")
    if "summaryBaseURL" not in updated or updated["summaryBaseURL"] in (
            "", "http://127.0.0.1:8000/v1"):
        updated["summaryBaseURL"] = "http://127.0.0.1:8000/v1"
    encoded = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode()
    backup = None
    if original is not None:
        fd, name = tempfile.mkstemp(prefix="config-before-runtime-", suffix=".json", dir=CONFIG.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(original)
        backup = Path(name)
    fd, name = tempfile.mkstemp(prefix="config-", suffix=".json", dir=CONFIG.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        if (CONFIG.read_bytes() if CONFIG.exists() else None) != original:
            raise RuntimeError("Lalia config changed during setup; no settings were replaced")
        staged.replace(CONFIG)
    finally:
        staged.unlink(missing_ok=True)
    if backup:
        print(f"Previous Lalia config backed up to {backup}")


def runtime_process_generation(pid, runtime_prefix, runtime_family):
    opened = subprocess.run(["/usr/sbin/lsof", "-p", pid, "-Fn"],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    paths = [Path(line[1:]).resolve() for line in opened if line.startswith("n")]
    if any(path.is_relative_to(runtime_prefix) for path in paths):
        return "current"
    if any(path.is_relative_to(runtime_family) for path in paths):
        return "older"
    return "foreign"


def check():
    try:
        port_pids = subprocess.run(["/usr/sbin/lsof", "-nP", "-iTCP:8002", "-sTCP:LISTEN", "-t"],
                                   check=True, capture_output=True, text=True).stdout.split()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("No Lalia runtime is listening on 127.0.0.1:8002; run `brew services start lalia-runtime`.") from exc
    runtime_prefix = str(Path(sys.prefix).resolve())
    runtime_family = str(Path(runtime_prefix).parent.parent)
    for pid in set(port_pids):
        generation = runtime_process_generation(pid, runtime_prefix, runtime_family)
        if generation != "current":
            if generation == "foreign":
                raise PortConflictError(f"Port 8002 is owned by another process (PID {pid}); refusing to restart or configure Lalia")
            raise RuntimeError("An older Lalia runtime owns port 8002 and needs a restart")
    try:
        models = request("/models")["data"]
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        raise RuntimeError("Runtime is not responding at 127.0.0.1:8002; run `brew services start lalia-runtime`.") from exc
    if MODEL not in {item.get("id") for item in models}:
        raise RuntimeError(f"{MODEL} is not loaded; run `lalia-runtime setup`.")
    print(f"{MODEL} available at {BASE_URL}")


def validate_inference_result(result, duration):
    """Check the MOSS response shape consumed by Lalia's transcript parser."""
    text = result.get("text")
    usage = result.get("usage")
    segments = result.get("segments")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("MOSS transcription smoke request returned empty text")
    if (not isinstance(usage, dict) or usage.get("type") != "tokens"
            or type(usage.get("input_tokens")) is not int or usage["input_tokens"] <= 0
            or type(usage.get("output_tokens")) is not int or not 0 <= usage["output_tokens"] < 1024
            or type(usage.get("total_tokens")) is not int
            or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]):
        raise RuntimeError("MOSS smoke response lacks valid token usage required by Lalia")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("MOSS smoke response lacks timed segments required by Lalia")
    pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        if len(matches) != len(segments):
            raise RuntimeError("MOSS smoke response text and timed segments do not match")
        cursor = 0
        for match, segment in zip(matches, segments):
            if text[cursor:match.start()].strip():
                raise RuntimeError("MOSS smoke response contains text outside timed segments")
            start, end = segment.get("start"), segment.get("end")
            segment_text = segment.get("text", "").strip()
            speaker = segment.get("speaker_id")
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or float(match.group(1)) != start or float(match.group(4)) != end
                    or speaker != match.group(2)
                    or segment_text != f"[{speaker}] {match.group(3).strip()}"
                    or start < 0 or end <= start or end > duration + 0.1):
                raise RuntimeError("MOSS smoke response contains invalid or mismatched timed segments")
            cursor = match.end()
        if text[cursor:].strip():
            raise RuntimeError("MOSS smoke response contains text outside timed segments")
    elif (len(segments) != 1 or segments[0].get("text") != text
          or segments[0].get("speaker_id") is not None
          or segments[0].get("start") != 0
          or type(segments[0].get("end")) not in (int, float)
          or not 0 < segments[0]["end"] <= duration + 0.1
          or re.search(r"\[(?:-?\d|S(?:\d|\]|$))", text)):
        raise RuntimeError("MOSS smoke response is not compatible with Lalia transcript parsing")


def verify_inference():
    """Exercise the installed model before changing the Lalia configuration."""
    boundary = "lalia-runtime-smoke"
    with tempfile.TemporaryDirectory(prefix="lalia-runtime-smoke-") as temporary:
        audio_path = Path(temporary) / "smoke.wav"
        subprocess.run(["/usr/bin/say", "-o", str(audio_path), "--data-format=LEI16@16000",
                        "Lalia runtime verification."], check=True,
                       capture_output=True, text=True)
        audio = audio_path.read_bytes()
        with wave.open(str(audio_path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
    parts = [("model", MODEL.encode()), ("language", b"en"), ("max_tokens", b"1024"),
             ("stream", b"false"), ("prompt", OFFICIAL_PROMPT.encode())]
    body = bytearray()
    for name, value in parts:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        body.extend(value)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"smoke.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    body.extend(audio)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(BASE_URL + "/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with opener.open(req, timeout=120) as response:
            result = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"MOSS transcription smoke request failed: {exc}") from exc
    validate_inference_result(result, duration)


def setup():
    provision_model()
    try:
        check()
    except PortConflictError:
        raise
    except RuntimeError:
        subprocess.run(["brew", "services", "restart", "lalia-runtime"], check=True)
        last_error = None
        ready = False
        for _ in range(60):
            try:
                check()
                ready = True
                break
            except RuntimeError as exc:
                last_error = exc
                time.sleep(2)
        if not ready:
            raise last_error or RuntimeError("Lalia runtime did not become ready")
    verify_inference()
    write_config()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(prog="lalia-runtime")
    parser.add_argument("command", choices=("setup", "check", "serve"))
    command = parser.parse_args().command
    if command == "setup":
        setup()
    elif command == "check":
        check()
    else:
        os.execv(sys.executable, [sys.executable, "-m", "omlx.cli", "serve",
                                  "--base-path", str(ROOT / "config"),
                                  "--model-dir", str(ROOT / "models"),
                                  "--host", "127.0.0.1", "--port", "8002",
                                  "--max-audio-upload-size", "250MB"])


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"lalia-runtime: {exc}", file=sys.stderr)
        raise SystemExit(1)
