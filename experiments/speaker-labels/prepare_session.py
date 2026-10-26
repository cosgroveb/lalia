"""Derive local evaluation audio from finalized Lalia segments without altering originals."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


RATE = 16_000


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def statistics(audio):
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    windows = audio[:len(audio) // 1600 * 1600].reshape(-1, 1600)
    quiet = np.mean(windows.astype(np.float64) ** 2, axis=1) < 1e-6
    return {
        "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "peak": float(np.max(np.abs(audio))),
        "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
        "quiet_100ms_fraction_below_minus60_dbfs": float(np.mean(quiet)),
    }


def encode(audio, destination):
    temporary = destination.with_suffix(".partial.flac")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le",
         "-ar", str(RATE), "-ac", "1", "-i", "pipe:0", "-c:a", "flac",
         "-sample_fmt", "s32", str(temporary)],
        input=audio.astype("<f4").tobytes(), check=True,
    )
    info = probe(temporary)
    if int(info["sample_rate"]) != RATE or int(info["channels"]) != 1:
        raise ValueError("Derived audio has the wrong format")
    if abs(float(info["duration"]) - len(audio) / RATE) > 1 / RATE:
        raise ValueError("Derived audio has the wrong duration")
    temporary.chmod(0o600)
    temporary.replace(destination)


def reconstruct(session, segments, frames):
    audio = np.zeros(frames, dtype=np.float32)
    previous_end = 0
    for segment in sorted(segments, key=lambda item: item["start"]):
        name = segment["filename"]
        path = session / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe or missing source file")
        info = probe(path)
        duration = segment["frames"] / segment["sampleRate"]
        if (int(info["sample_rate"]) != segment["sampleRate"]
                or int(info["channels"]) != segment["channels"]
                or abs(float(info["duration"]) - duration) > 2 / segment["sampleRate"]):
            raise ValueError("Source format or duration differs from the manifest")
        decoded = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
             "-f", "f32le", "-ac", "1", "-ar", str(RATE), "pipe:1"],
            capture_output=True, check=True,
        )
        samples = np.frombuffer(decoded.stdout, dtype="<f4")
        expected = round(duration * RATE)
        if abs(len(samples) - expected) > 1 or not np.isfinite(samples).all():
            raise ValueError("Unexpected decoded sample count or nonfinite audio")
        start = round(segment["start"] * RATE)
        end = start + len(samples)
        if start < previous_end - 1 or end > frames + 1:
            raise ValueError("Source segments overlap or exceed the timeline")
        # Rounding source timestamps to 16kHz can differ by one sample.
        end = min(end, frames)
        audio[start:end] = samples[:end - start]
        previous_end = end
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.chmod(0o700)
    record = json.loads((args.session / "session.json").read_text())
    if record["captureState"] == "recording":
        raise ValueError("Recording is still active")
    segments = record["segments"]
    if not segments or not all(item["complete"] for item in segments):
        raise ValueError("Every source segment must be finalized")
    duration = max(item["start"] + item["frames"] / item["sampleRate"] for item in segments)
    frames = round(duration * RATE)
    tracks = {}
    stats = {}
    for source in ("microphone", "system"):
        source_segments = [item for item in segments if item["source"] == source]
        if not source_segments:
            raise ValueError(f"Missing {source} track")
        tracks[source] = reconstruct(args.session, source_segments, frames)
        stats[source] = statistics(tracks[source])
        encode(tracks[source], args.output / f"{source}.flac")
    mixed = (tracks["microphone"] + tracks["system"]) * 0.5
    stats["mixed"] = statistics(mixed)
    encode(mixed, args.output / "mixed.flac")
    selected = "mixed"
    for source, other in (("microphone", "system"), ("system", "microphone")):
        rms = stats[source]["rms_dbfs"]
        if rms is None or rms < -70:
            selected = other
    selected_audio = mixed if selected == "mixed" else tracks[selected]
    short = selected_audio[:180 * RATE]
    encode(short, args.output / "short.flac")
    manifest = {
        "private": True,
        "source_session": str(args.session.resolve()),
        "source_capture_state": record["captureState"],
        "source_segments": len(segments),
        "sample_rate": RATE,
        "channels": 1,
        "selected_track": selected,
        "mix": "Equal-weight microphone and system, aligned by manifest start, gaps remain silence",
        "statistics": stats,
        "clips": [
            {"id": "lalia-short", "audio_path": str((args.output / "short.flac").resolve()),
             "duration": len(short) / RATE, "reference": None, "private": True},
            {"id": "lalia-full", "audio_path": str((args.output / f"{selected}.flac").resolve()),
             "duration": frames / RATE, "reference": None, "private": True},
        ],
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps({"manifest": str(path), "duration": frames / RATE,
                      "selected_track": selected, "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
