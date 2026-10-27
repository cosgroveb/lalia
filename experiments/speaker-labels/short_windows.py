#!/usr/bin/env python3
"""One bounded, public-only three-window MOSS diagnostic. No production writes."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request
import wave

from window_matching import (MODEL_NAME, MODEL_SHA256, RATE, THRESHOLD, clean_spans,
                             embed, footprint, interrupt, nearest, normalize, save,
                             sha256, stop)

ROOT = Path.home() / ".local/state/lalia-speaker-qualification/official-api-path"
EMBED_ROOT = ROOT.parent / "window-matching-20261027"
PUBLIC = Path.home() / ".local/state/lalia-speaker-eval-20261025/data/public/manifest.json"
LAUNCHER = Path("/tmp/unified-coordinator-lalia-speaker-labels/omlx-official-api")
MODEL = "MOSS-Transcribe-Diarize"
PROMPT_SHA = "5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a"
TOKENS = 49152
PORT = 18001


def speech(segments):
    from score import NON_SPEECH_EVENTS, clean_text

    return [s for s in segments if clean_text(s["text"]).strip() not in NON_SPEECH_EVENTS]


def embedding_clips(segments):
    """Use predicted single-label spans only, longest first, without references."""
    spans = clean_spans(segments)
    selected = {}
    for speaker in sorted({s["speaker_id"] for s in segments}):
        pool = []
        for span in spans:
            if span["speaker_id"] != speaker:
                continue
            start = math.ceil((span["start"] + 0.15) * RATE)
            end = math.floor((span["end"] - 0.15) * RATE)
            if end - start >= 3 * RATE:
                pool.append((start, end))
        # One 3s clip at each span's trimmed start, up to two distinct spans.
        pool.sort(key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
        if not pool:
            raise ValueError(f"No clean 3s predicted span for {speaker}")
        selected[speaker] = [{"start_frame": a, "end_frame": a + 3 * RATE}
                             for a, _ in pool[:2]]
    if not selected:
        raise ValueError("No predicted speech groups")
    return selected


def match_groups(groups, centroids):
    details, mapping = [], {}
    for local, vector in sorted(groups.items()):
        best, assigned, scores = nearest(vector, centroids)
        details.append({"local": local, "nearest": best, "assigned": assigned,
                        "scores": scores, "rejected": assigned is None})
        if assigned is not None:
            mapping[local] = assigned
    collisions = {global_id: sorted(local for local, value in mapping.items() if value == global_id)
                  for global_id, count in Counter(mapping.values()).items() if count > 1}
    return {"mapping": mapping, "assignments": details, "collisions": collisions,
            "accepted": len(mapping) == len(groups) and not collisions}


def offset_segments(segments, offset, window, mapping):
    return [dict(s, start=s["start"] + offset, end=s["end"] + offset,
                 speaker_id=mapping.get(s.get("speaker_id"), s.get("speaker_id")),
                 source={"window": window, "segment": index,
                         "local_speaker": s.get("speaker_id"), "offset_seconds": offset})
            for index, s in enumerate(segments)]


def shared_gap(left, right, lower, upper, nominal):
    occupied = sorted((max(lower, s["start"]), min(upper, s["end"]))
                      for s in left + right if s["start"] < upper and s["end"] > lower)
    cursor, gaps = lower, []
    for start, end in occupied + [(upper, upper)]:
        if start - cursor >= 0.2 - 1e-9:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if not gaps:
        raise ValueError("No shared speech gap of at least 0.2s")
    start, end = min(gaps, key=lambda gap: (abs(sum(gap) / 2 - nominal), gap[0]))
    return {"start": start, "end": end, "cut": (start + end) / 2, "nominal": nominal}


def join(windows, cuts, duration):
    from score import NON_SPEECH_EVENTS, clean_text

    kept, audit = [], []
    boundaries = [0, *cuts, duration]
    for index, segments in enumerate(windows):
        lower, upper = boundaries[index:index + 2]
        for segment in segments:
            event = clean_text(segment["text"]).strip() in NON_SPEECH_EVENTS
            if event:
                # An event can cross a speech-free seam. Keep it whole by midpoint.
                midpoint = (segment["start"] + segment["end"]) / 2
                retained = lower <= midpoint < upper or midpoint == duration == upper
                reason = "event midpoint ownership"
            else:
                retained = lower <= segment["start"] and segment["end"] <= upper
                redundant = segment["end"] <= lower or segment["start"] >= upper
                if not retained and not redundant:
                    raise ValueError("Speech segment crosses ownership cut")
                reason = "owned speech" if retained else "other window owns entire interval"
            audit.append({**segment["source"], "start": segment["start"], "end": segment["end"],
                          "event": event, "retained": retained, "reason": reason})
            if retained:
                kept.append(segment)
    return sorted(kept, key=lambda s: (s["start"], s["source"]["window"], s["source"]["segment"])), audit


def extract(source, output, start, end):
    with wave.open(str(source)) as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (RATE, 1, 2, "NONE"):
            raise ValueError("Expected uncompressed 16kHz mono PCM16")
        if not 0 <= start < end <= reader.getnframes():
            raise ValueError("Invalid frame interval")
        reader.setpos(start)
        pcm = reader.readframes(end - start)
        if len(pcm) != (end - start) * 2:
            raise ValueError("Short source read")
        with wave.open(str(output), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(pcm)
    with wave.open(str(output)) as check:
        if check.readframes(check.getnframes()) != pcm:
            raise ValueError("PCM extraction changed samples")
    return {"start_frame": start, "end_frame": end, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "wav_sha256": sha256(output), "audio_path": str(output), "duration": (end - start) / RATE}


def embedding_worker(state):
    import numpy as np
    import onnxruntime as ort

    model = EMBED_ROOT / MODEL_NAME
    if sha256(model) != MODEL_SHA256:
        raise ValueError("CAM++ model hash differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    protocol = json.loads((state / "protocol.json").read_text())
    selections = json.loads((state / "embedding-clips.json").read_text())
    centroids, matches = {}, []
    for index, (window, groups) in enumerate(zip(protocol["windows"], selections)):
        vectors = {}
        with wave.open(window["audio_path"]) as source:
            for local, clips in groups.items():
                embeddings = []
                for clip in clips:
                    source.setpos(clip["start_frame"])
                    embeddings.append(embed(source.readframes(clip["end_frame"] - clip["start_frame"]), session))
                vectors[local] = normalize(np.mean(embeddings, axis=0))
        if index == 0:
            centroids = {f"S{i + 1:02d}": vector for i, (_, vector) in enumerate(sorted(vectors.items()))}
            result = {"mapping": dict(zip(sorted(vectors), centroids)), "accepted": True,
                      "assignments": [], "collisions": {}, "enrollment": True}
        else:
            result = match_groups(vectors, centroids)
        matches.append(result)
        save(state / "assignments.json", matches)
        if not result["accepted"]:
            raise ValueError(f"Window {index} identity rejected or collided")
    save(state / "centroids.json", {label: vector.tolist() for label, vector in centroids.items()})


def reference_alignment(clip, response):
    import jiwer
    from score import by_speaker, cp_word_error_rate, normalize as words

    reference, hypothesis = clip["reference_segments"], speech(response["segments"])
    assignment = cp_word_error_rate(by_speaker(reference), by_speaker(hypothesis),
                                    reference_sort=False, hypothesis_sort=False).assignment
    statuses, insertions, unplaced = {}, [], 0
    for ref_id, hyp_id in assignment:
        refs = sorted((s for s in reference if s["speaker_id"] == ref_id), key=lambda s: s["start"])
        tokens, times = [], []
        for segment in refs:
            normalized = words(segment["text"]).split()
            tokens.extend(normalized)
            times.extend([(segment["start"] + segment["end"]) / 2] * len(normalized))
        hyp = by_speaker(hypothesis).get(hyp_id, "")
        alignment = jiwer.process_words(" ".join(tokens), hyp)
        for chunk in alignment.alignments[0]:
            for index in range(chunk.ref_start_idx, chunk.ref_end_idx):
                statuses[(ref_id, index)] = (times[index], chunk.type)
            if chunk.type == "insert":
                count = chunk.hyp_end_idx - chunk.hyp_start_idx
                if times:
                    index = min(chunk.ref_start_idx, len(times) - 1)
                    insertions.extend([times[index]] * count)
                else:
                    unplaced += count
    return statuses, insertions, unplaced


def compare(clip, response, baseline_response, baseline_result, cuts):
    from score import NON_SPEECH_EVENTS, clean_text, score

    scoring_segments = []
    for segment in response["segments"]:
        text = clean_text(segment["text"]).strip()
        scoring_segments.append(dict(segment, text=text if text in NON_SPEECH_EVENTS else segment["text"]))
    metrics = score(clip, dict(response, segments=scoring_segments), expect_speakers=True)
    previous = baseline_result["metrics"]
    deltas = {
        "cpwer": metrics["speaker_attributed"]["cpwer"] - previous["speaker_attributed"]["cpwer"],
        "strict_der": metrics["diarization"]["strict"]["der"] - previous["diarization"]["strict"]["der"],
        "collared_der": metrics["diarization"]["boundary_250ms"]["der"] - previous["diarization"]["boundary_250ms"]["der"],
    }
    current, inserts, unplaced = reference_alignment(clip, response)
    old, old_inserts, old_unplaced = reference_alignment(clip, baseline_response)
    seams = []
    for cut in cuts:
        nearby = {key for key, (at, _) in current.items() if abs(at - cut) <= 30}
        seams.append({"cut": cut, "radius_seconds": 30, "reference_words": len(nearby),
                      "merged_deletions": sum(current[key][1] == "delete" for key in nearby),
                      "baseline_deletions": sum(old[key][1] == "delete" for key in nearby),
                      "new_deletions_vs_baseline": sum(current[key][1] == "delete" and old[key][1] != "delete" for key in nearby),
                      "recovered_baseline_deletions": sum(current[key][1] != "delete" and old[key][1] == "delete" for key in nearby),
                      "merged_insertions": sum(abs(at - cut) <= 30 for at in inserts),
                      "baseline_insertions": sum(abs(at - cut) <= 30 for at in old_inserts)})
    return {"metrics": metrics, "diagnostic_regression": {"delta": deltas, "limit": 0.05,
            "passed": deltas["cpwer"] <= 0.05 and deltas["collared_der"] <= 0.05},
            "seams": seams, "unplaced_insertions": {"merged": unplaced, "baseline": old_unplaced},
            "alignment_limit": "Full-transcript speaker-permuted word alignment anchors insertions at the next reference word (last at end). Overlap ordering and repeated words can shift alignment. Insertions are not proven duplicates."}


def prepare(state):
    from qualification import baseline_metrics, completion

    state.mkdir(parents=True, exist_ok=True)
    if any(state.iterdir()):
        raise ValueError("Preparation requires a fresh state directory")
    if not state.is_relative_to(ROOT):
        raise ValueError("State must be under isolated official-api-path")
    manifest = json.loads(PUBLIC.read_text())
    clips = manifest["clips"] if isinstance(manifest, dict) else manifest
    clip, = [clip for clip in clips if clip["id"] == "ami-ES2004a-full" and not clip.get("private")]
    baseline_dir = ROOT / "baseline/results" / MODEL / clip["id"]
    baseline = json.loads((baseline_dir / "result.json").read_text())
    prompt = (ROOT / "official-prompt.txt").read_text()
    if sha256(ROOT / "official-prompt.txt") != PROMPT_SHA:
        raise ValueError("Prompt pin differs")
    baseline_metrics(clip, baseline, prompt)
    completion(json.loads((baseline_dir / "response.json").read_text()), clip["duration"], TOKENS)
    evidence = json.loads((ROOT / "qualification-code.json").read_text())["hashes"]
    runtime_pins = {path: digest for path, digest in evidence.items()
                    if str(ROOT / "runtime") in path or path == str(LAUNCHER) or path.endswith("official-api-runtime.py")}
    preflight = json.loads((ROOT / "preflight.json").read_text())
    model_pins = {path: digest for path, digest in preflight["hashes"].items() if "/baseline/models/" in path}
    runtime_model = ROOT / "runtime/mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py"
    runtime_pins[str(runtime_model)] = next(digest for path, digest in preflight["hashes"].items()
                                           if path.endswith("/moss_transcribe_diarize.py"))
    for path, expected in {**runtime_pins, **model_pins}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Pinned source/model changed: {path}")
    if sha256(EMBED_ROOT / MODEL_NAME) != MODEL_SHA256:
        raise ValueError("CAM++ pin differs")
    source = Path(clip["audio_path"])
    expected_audio = next(digest for path, digest in preflight["hashes"].items() if path.endswith("/ami-ES2004a-full.wav"))
    if sha256(source) != expected_audio:
        raise ValueError("Public audio differs from baseline fixture")
    with wave.open(str(source)) as audio:
        frames = audio.getnframes()
    if frames / RATE != clip["duration"] or not 750 < frames / RATE < 1100:
        raise ValueError("Unexpected public fixture duration")
    windows = []
    for index, (start, end) in enumerate(((0, 390 * RATE), (330 * RATE, 750 * RATE), (690 * RATE, frames))):
        directory = state / f"window-{index}"
        directory.mkdir()
        windows.append(extract(source, directory / "audio.wav", start, end))
    models = state / "models"
    models.mkdir()
    (models / MODEL).symlink_to((ROOT / "baseline/models" / MODEL).resolve(strict=True), target_is_directory=True)
    invariant_paths = [Path(path) for path in preflight["hashes"] if path.startswith(("/opt/homebrew/", "/Applications/", str(Path.home() / ".omlx/"), str(Path.home() / ".config/lalia/")))]
    invariant_paths += [source, PUBLIC, ROOT / "official-prompt.txt", baseline_dir / "response.json", baseline_dir / "result.json"]
    invariant_paths += list(map(Path, runtime_pins)) + list(map(Path, model_pins))
    invariant_paths += [Path(__file__).with_name(name).resolve()
                        for name in ("short_windows.py", "window_matching.py", "qualification.py", "score.py")]
    protocol = {"clip": clip["id"], "manifest": str(PUBLIC), "baseline": str(baseline_dir),
                "baseline_role": "Saved diagnostic comparator, not a new release qualification baseline",
                "windows": windows, "nominal_cuts": [360, 720], "overlaps": [[330, 390], [690, 750]],
                "prompt_sha256": PROMPT_SHA, "max_tokens": TOKENS, "threshold": THRESHOLD,
                "model_revision": preflight["model_revision"], "launcher": str(LAUNCHER),
                "embedding": "Predicted single-label spans, trim 150ms, up to two longest distinct spans, first 3s each. Frozen first-window normalized means. Reject below 0.55 or collisions.",
                "seam": "Nearest shared >=0.2s speech-free gap midpoint. Whole speech segments only. Events retained whole by midpoint ownership.",
                "guards": {"request_seconds": 300, "overall_seconds": 600, "footprint_gib": 16, "battery_percent": 25},
                "invariants": {str(path): sha256(path) for path in invariant_paths}}
    save(state / "protocol.json", protocol)
    return protocol


def power():
    value = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True, timeout=5).stdout
    percent = re.search(r"(\d+)%", value)
    if "Battery Power" in value and (not percent or int(percent[1]) <= 25):
        raise RuntimeError("Battery cutoff: require >25% or AC power")
    return value


def deadline(*_):
    raise TimeoutError("Overall 600s deadline")


class Guard:
    def __init__(self, state):
        self.state, self.started, self.children, self.samples = state, time.monotonic(), [], []
        self.last_power = -10

    def launch(self, command, log, **kwargs):
        with log.open("w") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, **kwargs)
        self.children.append(child)
        save(self.state / "processes.json", {"runner": os.getpid(), "children": [p.pid for p in self.children]})
        return child

    def check(self):
        elapsed = time.monotonic() - self.started
        if elapsed >= 600:
            raise TimeoutError("Overall 600s deadline")
        active = [child for child in self.children if child.poll() is None]
        used = 0
        for child in active:
            try:
                used += footprint(child.pid)
            except ProcessLookupError:
                if child.poll() is None:
                    raise
        self.samples.append({"seconds": elapsed, "footprint_bytes": used})
        if used > 16 * 1024**3:
            raise MemoryError("Combined child footprint exceeded 16GiB")
        if elapsed - self.last_power >= 10:
            power()
            self.last_power = elapsed

    def wait(self, child, timeout, server=None):
        until = time.monotonic() + timeout
        while child.poll() is None:
            self.check()
            if time.monotonic() >= until:
                raise TimeoutError("Child deadline")
            if server is not None and server.poll() is not None:
                raise RuntimeError("Isolated server exited")
            time.sleep(0.5)
        child.wait()
        if child.returncode:
            raise RuntimeError(f"Child {child.pid} exited {child.returncode}, see local log")

    def close(self):
        for child in reversed(self.children):
            stop(child)
        save(self.state / "memory.json", self.samples)
        save(self.state / "cleanup.json", {"children": [{"pid": p.pid, "returncode": p.returncode} for p in self.children]})


def execute(state):
    from qualification import completion

    protocol = json.loads((state / "protocol.json").read_text())
    if (state / "started.json").exists():
        raise ValueError("Run already attempted, no automatic retry")
    for path, expected in protocol["invariants"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Preflight invariant changed: {path}")
    for window in protocol["windows"]:
        if sha256(Path(window["audio_path"])) != window["wav_sha256"]:
            raise ValueError("Extracted audio changed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    power_before = power()
    guard = Guard(state)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    result = {"status": "failed", "scope": "Public AMI short-window closed-set diagnostic"}
    try:
        save(state / "started.json", {"pid": os.getpid(), "unix_seconds": time.time(), "power": power_before})
        guard.launch(["caffeinate", "-i", "-w", str(os.getpid())], state / "awake.log")
        server_dir = state / "server"
        server_dir.mkdir()
        command = [str(LAUNCHER), "serve", "--base-path", str(server_dir), "--model-dir", str(state / "models"),
                   "--host", "127.0.0.1", "--port", str(PORT), "--no-hf-cache", "--no-cache", "--headless",
                   "--max-concurrent-requests", "1", "--max-audio-upload-size", "250MB", "--memory-guard", "safe", "--log-level", "warning"]
        server = guard.launch(command, state / "server.log", cwd=server_dir,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        started = time.monotonic()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            guard.check()
            if server.poll() is not None or time.monotonic() - started >= 120:
                raise RuntimeError("Isolated server failed readiness")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=1) as response:
                    if MODEL in {entry["id"] for entry in json.load(response)["data"]}:
                        break
            except OSError:
                pass
            time.sleep(0.5)
        replies, selections, request_results = [], [], []
        prompt = (ROOT / "official-prompt.txt").read_text()
        for index, window in enumerate(protocol["windows"]):
            guard.check()
            directory = state / f"window-{index}"
            command = ["curl", "--silent", "--show-error", "--fail-with-body", "--noproxy", "*", "--max-time", "300",
                       f"http://127.0.0.1:{PORT}/v1/audio/transcriptions", "--form", f"file=@{window['audio_path']}",
                       "--form", f"model={MODEL}", "--form", "language=en", "--form", "stream=false",
                       "--form", f"max_tokens={TOKENS}", "--form-string", f"prompt={prompt}",
                       "--output", str(directory / "response.json")]
            started = time.monotonic()
            client = guard.launch(command, directory / "client.log")
            guard.wait(client, 300, server)
            response = json.loads((directory / "response.json").read_text())
            checked = completion(response, window["duration"], TOKENS)
            request_results.append({"window": index, "wall_seconds": time.monotonic() - started, "completion": checked})
            save(state / "requests.json", request_results)
            selections.append(embedding_clips(speech(response["segments"])))
            replies.append(response)
            print(f"Window {index}: completion valid", flush=True)
        stop(server)
        save(state / "embedding-clips.json", selections)
        worker = guard.launch([str(EMBED_ROOT / "venv/bin/python"), "-B", str(Path(__file__).resolve()),
                               "--state", str(state), "--embed-worker"], state / "embedding.log",
                              env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"})
        guard.wait(worker, 60)
        assignments = json.loads((state / "assignments.json").read_text())
        windows = [offset_segments(reply["segments"], window["start_frame"] / RATE, index, assignment["mapping"])
                   for index, (reply, window, assignment) in enumerate(zip(replies, protocol["windows"], assignments))]
        seams = [shared_gap(speech(windows[i]), speech(windows[i + 1]), *overlap, nominal)
                 for i, (overlap, nominal) in enumerate(zip(protocol["overlaps"], protocol["nominal_cuts"]))]
        save(state / "seams.json", seams)
        duration = protocol["windows"][-1]["end_frame"] / RATE
        merged, audit = join(windows, [seam["cut"] for seam in seams], duration)
        save(state / "segment-accounting.json", audit)
        save(state / "merged-response.json", {"segments": merged, "text": ""})
        guard.check()
        manifest = json.loads(PUBLIC.read_text())
        clips = manifest["clips"] if isinstance(manifest, dict) else manifest
        clip, = [clip for clip in clips if clip["id"] == protocol["clip"]]
        baseline = Path(protocol["baseline"])
        result.update(compare(clip, {"segments": merged}, json.loads((baseline / "response.json").read_text()),
                              json.loads((baseline / "result.json").read_text()), [seam["cut"] for seam in seams]))
        guard.check()
        result.update(status="complete", assignments=assignments, join_seams=seams,
                      accounting=dict(Counter(("event" if row["event"] else "speech") + ("_kept" if row["retained"] else "_redundant") for row in audit)))
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        guard.close()
        result["wall_seconds"] = time.monotonic() - guard.started
        result["peak_sampled_footprint_gib"] = max((row["footprint_bytes"] for row in guard.samples), default=0) / 1024**3
        result["invariants_unchanged"] = all(sha256(Path(path)) == expected for path, expected in protocol["invariants"].items())
        save(state / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--embed-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    state = args.state.resolve()
    if args.prepare:
        prepare(state)
    elif args.embed_worker:
        embedding_worker(state)
    else:
        execute(state)


if __name__ == "__main__":
    main()
