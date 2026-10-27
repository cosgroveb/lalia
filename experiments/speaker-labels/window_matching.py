#!/usr/bin/env python3
"""Bounded CPU CAM++ feasibility on disjoint, reference-selected AMI speech."""

import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import wave

RATE = 16000
THRESHOLD = 0.55
MODEL_NAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
MODEL_SHA256 = "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11"
MODEL_REVISION = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
MODEL_URL = f"https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/{MODEL_REVISION}/{MODEL_NAME}"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def footprint(pid):
    # Same macOS rusage_info_v0 layout used by the neighboring ASR runner.
    buffer = ctypes.create_string_buffer(96)
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 0, buffer) != 0:
        raise ProcessLookupError(pid)
    return int.from_bytes(buffer.raw[72:80], "little")


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait()


def interrupt(*_):
    raise KeyboardInterrupt


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def clean_spans(turns):
    """Keep only intervals with exactly one reference speaker active."""
    events = {}
    for turn in turns:
        start, end = turn["start"], turn["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid reference interval")
        events.setdefault(start, []).append((turn["speaker_id"], 1))
        events.setdefault(end, []).append((turn["speaker_id"], -1))
    spans, active, previous = [], Counter(), 0
    for at, changes in sorted(events.items()):
        speakers = [speaker for speaker, count in active.items() if count > 0]
        if len(speakers) == 1 and at > previous:
            speaker = speakers[0]
            if spans and spans[-1]["speaker_id"] == speaker and spans[-1]["end"] == previous:
                spans[-1]["end"] = at
            else:
                spans.append({"start": previous, "end": at, "speaker_id": speaker})
        for speaker, count in changes:
            active[speaker] += count
        previous = at
    return spans


def select_clips(clip):
    speakers = sorted({turn["speaker_id"] for turn in clip["reference_turns"]})
    if len(speakers) != 4:
        raise ValueError("Expected four reference speakers")
    candidates = []
    for index, span in enumerate(clean_spans(clip["reference_turns"])):
        start = math.ceil((span["start"] + 0.15) * RATE)
        end = math.floor((span["end"] - 0.15) * RATE)
        while start + 3 * RATE <= end:
            candidate = {"speaker_id": span["speaker_id"], "span": index,
                         "start_frame": start, "end_frame": start + 3 * RATE}
            # Check the manual word-level reference for conflicting speakers too.
            if not any(word["speaker_id"] != span["speaker_id"]
                       and word["start"] < candidate["end_frame"] / RATE
                       and word["end"] > start / RATE for word in clip["reference_segments"]):
                candidates.append(candidate)
            start += 4 * RATE
    selected, counts = [], {}
    for speaker in speakers:
        pool = [sample for sample in candidates if sample["speaker_id"] == speaker]
        enrollment, used_spans = [], set()
        for sample in pool:
            if sample["span"] not in used_spans and len(enrollment) < 2:
                enrollment.append(dict(sample, role="enrollment"))
                used_spans.add(sample["span"])
        if len(enrollment) != 2:
            raise ValueError(f"Insufficient enrollment spans for {speaker}")
        held_out = [sample for sample in pool
                    if sample["start_frame"] >= enrollment[-1]["end_frame"] + 30 * RATE]
        # Pre-inference amendment: MEO015 supplies only three eligible later clips.
        held_out_count = 3 if speaker == "MEO015" else 4
        if len(held_out) < held_out_count:
            raise ValueError(f"Insufficient later held-out clips for {speaker}: {len(held_out)}")
        evaluation = [dict(held_out[i * (len(held_out) - 1) // (held_out_count - 1)], role="evaluation")
                      for i in range(held_out_count)]
        selected.extend(enrollment + evaluation)
        counts[speaker] = {"candidate_clips": len(pool), "enrollment_clips": 2,
                           "evaluation_clips": held_out_count,
                           "enrollment_seconds": 6, "evaluation_seconds": 3 * held_out_count,
                           "minimum_separation_seconds":
                               (evaluation[0]["start_frame"] - enrollment[-1]["end_frame"]) / RATE}
    return selected, counts


def normalize(vector):
    import numpy as np

    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid embedding norm")
    return vector / norm


def nearest(embedding, centroids):
    """No reference label or held-out centroid update enters matching."""
    scores = {speaker: float(embedding @ centroid) for speaker, centroid in centroids.items()}
    speaker = max(scores, key=scores.get)
    return speaker, speaker if scores[speaker] >= THRESHOLD else None, scores


def ready(per_speaker):
    return len(per_speaker) == 4 and all(
        counts["correct_accepted"] >= 2 and counts["wrong_accepted"] == 0 and counts["rejected"] == 0
        for counts in per_speaker.values())


def embed(audio, session):
    # Adapted from OpenMOSS/MOSS-Transcribe-Diarize PR 58, speaker.py at
    # 309f8f6d3e5cc94cebdb4d5f6b3fb9765409c2a5, Apache-2.0.
    # Modified to validate fixed-length samples and use a caller-owned CPU session.
    # License: third-party/MOSS-LICENSE.txt. Preserve peak normalization and CMN.
    import kaldi_native_fbank as knf
    import numpy as np

    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32)
    if len(samples) != 3 * RATE or not np.any(samples):
        raise ValueError("Expected three seconds of non-silent PCM")
    samples /= np.max(np.abs(samples))
    options = knf.FbankOptions()
    options.frame_opts.samp_freq = RATE
    options.frame_opts.dither = 0
    options.frame_opts.frame_length_ms = 25
    options.frame_opts.frame_shift_ms = 10
    options.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(options)
    fbank.accept_waveform(RATE, samples)
    fbank.input_finished()
    features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)
    features -= features.mean(axis=0, keepdims=True)
    vector = session.run(None, {session.get_inputs()[0].name: features[None, :, :]})[0].reshape(-1)
    if vector.shape != (192,):
        raise ValueError("Expected 192-dimensional CAM++ embedding")
    return normalize(vector)


def worker(output):
    import numpy as np
    import onnxruntime as ort

    started = time.monotonic()
    protocol = json.loads((output / "protocol.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(protocol["model_path"], sess_options=options,
                                   providers=["CPUExecutionProvider"])
    vectors = []
    with wave.open(protocol["audio_path"]) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("Expected 16 kHz mono PCM16 input")
        for sample in protocol["clips"]:
            source.setpos(sample["start_frame"])
            vectors.append(embed(source.readframes(sample["end_frame"] - sample["start_frame"]), session))
    centroids = {}
    for speaker in protocol["counts"]:
        enrolled = [vector for sample, vector in zip(protocol["clips"], vectors)
                    if sample["role"] == "enrollment" and sample["speaker_id"] == speaker]
        centroids[speaker] = normalize(np.mean(enrolled, axis=0))
    confusion = {speaker: Counter() for speaker in centroids}
    threshold_confusion = {speaker: Counter() for speaker in centroids}
    genuine, impostor, details = [], [], []
    accepted, rejected, wrong, correct = Counter(), Counter(), Counter(), Counter()
    for sample, vector in zip(protocol["clips"], vectors):
        if sample["role"] != "evaluation":
            continue
        best, assigned, scores = nearest(vector, centroids)
        truth = sample["speaker_id"]
        confusion[truth][best] += 1
        threshold_confusion[truth][assigned or "rejected"] += 1
        genuine.append(scores[truth])
        impostor.extend(value for speaker, value in scores.items() if speaker != truth)
        if assigned is None:
            rejected[truth] += 1
        else:
            accepted[truth] += 1
            if assigned == truth:
                correct[truth] += 1
            else:
                wrong[truth] += 1
        details.append(dict(sample, nearest=best, assigned=assigned, scores=scores))
    per_speaker = {speaker: {"correct_accepted": correct[speaker], "wrong_accepted": wrong[speaker],
                             "rejected": rejected[speaker]} for speaker in centroids}
    result = {
        "scope": "Oracle-segment embedding feasibility, not diarization or open-set qualification",
        "threshold": THRESHOLD, "counts": protocol["counts"],
        "nearest_confusion": confusion, "threshold_confusion": threshold_confusion,
        "accepted": sum(accepted.values()), "rejected": sum(rejected.values()),
        "wrong_accepted": sum(wrong.values()), "correct_accepted": sum(correct.values()),
        "per_speaker": per_speaker,
        "genuine_cosine_range": [min(genuine), max(genuine)],
        "impostor_cosine_range": [min(impostor), max(impostor)],
        "genuine_below_threshold_split_candidates": sum(value < THRESHOLD for value in genuine),
        "impostor_above_threshold_merge_candidates": sum(value >= THRESHOLD for value in impostor),
        "passes": ready(per_speaker),
        "embedding_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("numpy", "onnxruntime", "kaldi-native-fbank")},
        "python": sys.version, "providers": session.get_providers(),
        "model_metadata": session.get_modelmeta().custom_metadata_map,
    }
    save(output / "clip-scores.json", details)
    save(output / "result.json", result)


def battery():
    power = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout
    match = re.search(r"(\d+)%", power)
    if not match:
        raise RuntimeError("Could not read battery charge")
    percent = int(match[1])
    if "Battery Power" in power and percent <= 25:
        raise RuntimeError("Battery reached 25%")
    return percent


def main():
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        worker(args.output)
        return
    if args.manifest is None or args.model is None:
        parser.error("--manifest and --model are required")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    clip = next(clip for clip in json.loads(args.manifest.read_text())["clips"]
                if clip["id"] == "ami-ES2004a-full")
    if clip["private"] is not False or clip.get("stress_fixture") or clip.get("repeated_blocks"):
        raise ValueError("Only the public, original AMI meeting is allowed")
    selected, counts = select_clips(clip)
    protocol = {
        "policy": "RTTM single-speaker spans, 150 ms trim, 3 s clips on a 4 s grid. "
                  "Manual words veto cross-speaker contamination. First two distinct spans enroll. "
                  "Four index-quantile held-out clips start >=30 s after final enrollment, "
                  "except MEO015 has three after an explicit pre-inference amendment. "
                  "Frozen normalized mean centroids, cosine >=0.55 accepts. "
                  "Pass requires every selected clip correctly accepted, zero rejects/wrong matches.",
        "pre_inference_amendment": "Initial four-clip selection failed: "
                                   "Insufficient later held-out clips for MEO015: 3. "
                                   "Retain all three, exceeding original minimum of two. "
                                   "All other selection rules unchanged. No scores observed.",
        "manifest_sha256": sha256(args.manifest), "audio_path": clip["audio_path"],
        "audio_sha256": sha256(Path(clip["audio_path"])), "model_path": str(args.model.resolve()),
        "model_url": MODEL_URL, "model_sha256": MODEL_SHA256,
        "clips": selected, "counts": counts, "threshold": THRESHOLD,
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing frozen protocol differs")
    save(path, protocol)
    if args.select_only:
        print(json.dumps({"counts": counts, "selected_audio_seconds": len(selected) * 3}, indent=2))
        return
    if (output / "result.json").exists():
        raise ValueError("Refusing to overwrite an existing result")
    if sha256(args.model) != MODEL_SHA256:
        raise ValueError("Model SHA256 mismatch")
    start_battery = battery()
    started, peak = time.monotonic(), 0
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "VECLIB_MAXIMUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    "--output", str(output), "--worker"],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        save(output / "process.json", {"pid": process.pid, "state": "running"})
        while process.poll() is None:
            try:
                peak = max(peak, footprint(process.pid))
            except ProcessLookupError:
                if process.poll() is None:
                    raise
            if peak > 1024**3:
                raise MemoryError("Worker exceeded 1 GiB physical footprint")
            if time.monotonic() - started > 300:
                raise TimeoutError("Probe exceeded five minutes")
            battery()
            time.sleep(0.5)
        if process.returncode:
            raise RuntimeError(f"Worker failed with exit {process.returncode}. See worker.log")
        result = json.loads((output / "result.json").read_text())
        result.update(wall_seconds=time.monotonic() - started, sampled_peak_footprint_bytes=peak,
                      footprint_sample_seconds=0.5, battery_start=start_battery, battery_end=battery())
        save(output / "result.json", result)
        print(json.dumps(result, indent=2))
    finally:
        stop(process)
        save(output / "process.json", {"pid": process.pid, "state": "exited", "exit_code": process.returncode})


if __name__ == "__main__":
    main()
