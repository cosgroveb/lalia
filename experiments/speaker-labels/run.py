"""Run serial local oMLX benchmarks. Raw responses stay in the private state directory."""

import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from score import score


GIB = 1024**3
MODELS = ["cohere-transcribe-03-2026", "MOSS-Transcribe-Diarize", "VibeVoice-ASR-bf16"]
LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint(pid):
    # rusage_info_v0: UUID followed by ten uint64 fields (sys/resource.h).
    buffer = ctypes.create_string_buffer(96)
    if LIBPROC.proc_pid_rusage(pid, 0, buffer) != 0:
        raise ProcessLookupError(pid)
    return int.from_bytes(buffer.raw[72:80], "little")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def interrupt(*_):
    raise KeyboardInterrupt


def start_server(args, model):
    server_dir = args.state / "server" / model
    model_dir = server_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    link = model_dir / model
    target = (args.state / "models" / model).resolve(strict=True)
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    command = [
        args.omlx, "serve", "--base-path", str(server_dir), "--model-dir", str(model_dir),
        "--host", "127.0.0.1", "--port", str(args.port), "--no-hf-cache", "--no-cache",
        "--headless", "--max-concurrent-requests", "1", "--max-audio-upload-size", "250MB",
        "--memory-guard", "safe", "--log-level", "warning",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    log = (server_dir / "server.log").open("a")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                              start_new_session=True, cwd=server_dir)
    log.close()
    save(args.state / "server-process.json", {"pid": server.pid, "model": model})
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=1) as response:
                    ids = {entry["id"] for entry in json.load(response)["data"]}
                if model in ids:
                    return server
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise TimeoutError("Server did not discover the requested model within120s")
    except BaseException:
        stop(server)
        raise


def transcribe(args, server, model, clip, directory):
    tokens = 1024 if model.startswith("cohere") else (8192 if clip["duration"] <= 180 else 32768)
    timeout = args.timeout
    if args.deadline:
        timeout = min(timeout, max(1, args.deadline - time.time()))
    command = [
        "curl", "--silent", "--show-error", "--fail-with-body", "--max-time", str(timeout),
        f"http://127.0.0.1:{args.port}/v1/audio/transcriptions",
        "--form", f"file=@{clip['audio_path']}", "--form", f"model={model}",
        "--form", "language=en", "--form", "stream=false", "--form", f"max_tokens={tokens}",
        "--output", str(directory / "response.json"),
        "--write-out", "%{time_total}\n",
    ]
    started = time.monotonic()
    measurements = []
    with (directory / "client.log").open("w") as log, (directory / "http-seconds.txt").open("w") as timing:
        request = subprocess.Popen(command, stdout=timing, stderr=log, start_new_session=True)
        try:
            while request.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                used = footprint(server.pid)
                measurements.append({"seconds": round(time.monotonic() - started, 2), "footprint_bytes": used})
                if used > args.memory_gib * GIB:
                    raise MemoryError(f"Server exceeded {args.memory_gib}GiB physical footprint")
                if time.monotonic() - started > timeout + 5:
                    raise TimeoutError("Request exceeded deadline")
                time.sleep(0.5)
            if request.returncode:
                raise RuntimeError(f"HTTP client exited {request.returncode}. See local response/client log")
            elapsed = float((directory / "http-seconds.txt").read_text().strip())
            response = json.loads((directory / "response.json").read_text())
            metrics = score(clip, response, expect_speakers=not model.startswith("cohere"))
            return {
                "status": "complete", "wall_seconds": elapsed,
                "real_time_factor": elapsed / clip["duration"],
                "max_tokens": tokens, "metrics": metrics,
            }
        finally:
            stop(request)
            save(directory / "memory.json", measurements)


def report(state):
    rows = []
    for path in sorted((state / "results").glob("*/*/result.json")):
        rows.append(json.loads(path.read_text()))
    save(state / "summary.json", rows)
    lines = ["# Local speaker-ASR experiment", "", "Aggregate results. Raw audio and transcripts remain local.", "",
             "| Model | Clip | Status | Wall seconds | Peak footprint GiB |", "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['clip']} | {row['status']} | {row.get('wall_seconds', 0):.1f} | {row.get('peak_footprint_gib', 0):.2f} |")
    (state / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--omlx", default=shutil.which("omlx"))
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--memory-gib", type=float, default=64)
    parser.add_argument("--deadline", type=float, help="Unix timestamp, stop starting requests after this time")
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupt)
    args.state = args.state.resolve()
    clips = []
    for manifest in args.manifest:
        document = json.loads(manifest.read_text())
        clips.extend(document["clips"] if isinstance(document, dict) else document)
    if args.clips:
        clips = [clip for clip in clips if clip["id"] in args.clips]
    if not clips:
        raise ValueError("No clips selected")
    save(args.state / "runner-process.json", {"pid": os.getpid(), "started_at": dt.datetime.now().astimezone().isoformat()})
    server = None
    try:
        for model in args.models:
            try:
                for clip in clips:
                    directory = args.state / "results" / model / clip["id"]
                    directory.mkdir(parents=True, exist_ok=True)
                    if (directory / "result.json").exists():
                        continue
                    if args.deadline and time.time() >= args.deadline:
                        raise TimeoutError("Experiment deadline reached")
                    print(f"Running {model} / {clip['id']}", flush=True)
                    row = {"model": model, "clip": clip["id"], "audio_seconds": clip["duration"],
                           "private": clip.get("private", False), "started_at": dt.datetime.now().astimezone().isoformat(),
                           "cold_load": server is None}
                    save(args.state / "progress.json", {**row, "status": "running"})
                    try:
                        if server is None:
                            startup = time.monotonic()
                            server = start_server(args, model)
                            row["server_startup_seconds"] = time.monotonic() - startup
                        row.update(transcribe(args, server, model, clip, directory))
                    except (RuntimeError, OSError, ValueError, TimeoutError) as error:
                        row.update(status="failed", error=str(error))
                        print(f"Failed {model} / {clip['id']}: {error}", flush=True)
                        if server is not None:
                            stop(server)
                            server = None
                    except MemoryError as error:
                        row.update(status="resource_limit", error=str(error))
                        if server is not None:
                            stop(server)
                            server = None
                    memory = directory / "memory.json"
                    if memory.exists():
                        row["peak_footprint_gib"] = max((m["footprint_bytes"] for m in json.loads(memory.read_text())), default=0) / GIB
                    save(directory / "result.json", row)
                    report(args.state)
            finally:
                if server is not None:
                    stop(server)
                    server = None
        save(args.state / "progress.json", {"status": "finished", "finished_at": dt.datetime.now().astimezone().isoformat()})
    except BaseException as error:
        save(args.state / "progress.json", {"status": "interrupted", "error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        if server is not None:
            stop(server)
        report(args.state)


if __name__ == "__main__":
    main()
