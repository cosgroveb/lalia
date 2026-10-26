"""Validate pinned MOSS completion and report reference timing without text."""

import hashlib
import math
import re
import statistics

from score import by_speaker, clean_text, cp_word_error_rate, normalize


SEGMENT = re.compile(r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]"
                     r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]", re.DOTALL)


def completion(response, duration, max_tokens):
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("type") != "tokens":
        raise ValueError("Missing token usage")
    counts = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
    if (any(type(value) is not int for value in counts) or counts[0] <= 0
            or counts[1] < 0 or counts[2] != counts[0] + counts[1]):
        raise ValueError("Invalid token usage")
    if counts[0] + max_tokens > 131072:
        raise ValueError("Request exceeds model context")
    if counts[1] >= max_tokens:
        raise ValueError("Output token ceiling exhausted")
    text, segments = response.get("text"), response.get("segments") or []
    if not isinstance(text, str) or not isinstance(segments, list):
        raise ValueError("Malformed response")
    if counts[1] == 0 and (text.strip() or any(segment.get("text", "").strip() for segment in segments)):
        raise ValueError("Nonempty response with zero output tokens")
    if not text.strip() and all(not segment.get("text", "").strip() for segment in segments):
        return {"usage": usage, "eos": True, "raw_coverage": True, "segments": 0}
    if (len(segments) == 1 and not re.fullmatch(r"S\d+", str(segments[0].get("speaker_id", "")))
            and segments[0].get("text") == text
            and not re.search(r"\[(?:-?\d|S(?:\d|\]|$))", text)):
        start, end = segments[0].get("start"), segments[0].get("end")
        if (all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
                and 0 <= start < end <= duration):
            return {"usage": usage, "eos": True, "raw_coverage": True, "segments": 1}
    matches = list(SEGMENT.finditer(text))
    cursor = 0
    for match in matches:
        if text[cursor:match.start()].strip():
            raise ValueError("Unmatched raw transcript span")
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("Unmatched raw transcript span")
    if len(matches) != len(segments):
        raise ValueError("Raw/structured segment count differs")
    for match, segment in zip(matches, segments):
        start, end = segment.get("start"), segment.get("end")
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end))
                or not 0 <= start < end <= duration):
            raise ValueError("Invalid segment timestamps")
        if (start != float(match["start"]) or end != float(match["end"])
                or segment.get("speaker_id") != match["speaker"]
                or clean_text(segment.get("text", "")).strip() != match["text"].strip()):
            raise ValueError("Raw/structured segment content differs")
    return {"usage": usage, "eos": True, "raw_coverage": True, "segments": len(segments)}


def reference_timing(clip, response):
    words = clip.get("reference_segments", [])
    segments = response.get("segments") or []
    result = {}
    for name, fraction in (("start", 0), ("middle", 0.5), ("end", 1)):
        center = clip["duration"] * fraction
        errors = []
        for segment in segments:
            if abs(segment["start"] - center) > 120:
                continue
            needle = normalize(clean_text(segment["text"])).split()[:5]
            if len(needle) < 5:
                continue
            nearby = [word for word in words if abs(word["start"] - segment["start"]) < 90]
            for speaker in {word["speaker_id"] for word in nearby}:
                speaker_words = [word for word in nearby if word["speaker_id"] == speaker]
                tokens, starts = [], []
                for word in speaker_words:
                    normalized = normalize(word["text"]).split()
                    tokens.extend(normalized)
                    starts.extend([word["start"]] * len(normalized))
                matches = [i for i in range(len(tokens) - len(needle) + 1) if tokens[i:i + len(needle)] == needle]
                if len(matches) == 1:
                    errors.append(segment["start"] - starts[matches[0]])
        result[name] = {"matched_passages": len(errors),
                        "median_start_error_seconds": statistics.median(errors) if errors else None,
                        "median_absolute_start_error_seconds": statistics.median(map(abs, errors)) if errors else None,
                        "max_absolute_start_error_seconds": max(map(abs, errors)) if errors else None}
    return result


def baseline_metrics(clip, baseline, prompt=None):
    if baseline is None:
        if clip.get("stress_fixture") or clip.get("repeated_blocks") or clip["duration"] >= 3600:
            raise ValueError("Long or repeated qualification requires a successful short baseline")
        return None
    try:
        metrics = baseline["metrics"]
        cp = metrics["speaker_attributed"]
        der = metrics["diarization"]["boundary_250ms"]["der"]
        duration = baseline["audio_seconds"]
        valid = (baseline["status"] == "complete" and baseline["model"] == "MOSS-Transcribe-Diarize"
                 and baseline["completion"]["eos"] is True and baseline["completion"]["raw_coverage"] is True
                 and baseline["qualification"]["accepted"] is True and baseline["qualification"]["failures"] == []
                 and type(duration) in (int, float) and math.isfinite(duration) and 0 < duration < 3600
                 and metrics["schema"]["speaker_count"] == 4 and metrics["schema"]["missing_speaker_count"] == 0
                 and cp["missed_speakers"] == 0 and cp["extra_speakers"] == 0
                 and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                         for value in (cp["cpwer"], der)))
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Invalid or unsuccessful short baseline")
    if ("prompt" not in baseline or baseline["prompt"] != prompt
            or (prompt is not None and baseline.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest())):
        raise ValueError("Qualification requires a baseline with the same prompt")
    return metrics


def acceptance(clip, response, metrics, baseline=None, prompt=None):
    baseline = baseline_metrics(clip, baseline, prompt)
    schema = metrics["schema"]
    failures = []
    if schema["speaker_count"] != 4 or schema["missing_speaker_count"]:
        failures.append("Expected four globally assigned speakers")
    final_reference = schema.get("reference_last_word_end")
    final_hypothesis = schema.get("last_speech_end")
    if final_reference is not None and (final_hypothesis is None or final_hypothesis < final_reference - 10):
        failures.append("Missing final speech-bearing interval")
    cp = metrics.get("speaker_attributed")
    if not cp or cp["missed_speakers"] or cp["extra_speakers"]:
        failures.append("Global speaker assignment failed")
    continuity = []
    if clip.get("repeated_blocks") and not schema["missing_speaker_count"]:
        reference = clip["reference_segments"]
        hypothesis = response.get("segments") or []
        global_assignment = dict(cp_word_error_rate(by_speaker(reference), by_speaker(hypothesis),
                                                   reference_sort=False, hypothesis_sort=False).assignment)
        for block in clip["repeated_blocks"]:
            refs = [segment for segment in reference if block["start"] <= segment["start"] < block["end"]]
            hyps = [segment for segment in hypothesis if block["start"] <= segment["start"] < block["end"]]
            assignment = dict(cp_word_error_rate(by_speaker(refs), by_speaker(hyps),
                                                reference_sort=False, hypothesis_sort=False).assignment)
            agrees = all(speaker in global_assignment and global_assignment[speaker] == label
                         for speaker, label in assignment.items())
            continuity.append({"start": block["start"], "end": block["end"], "global_assignment_agrees": agrees,
                               "reference_speakers": len(by_speaker(refs)), "hypothesis_speakers": len(by_speaker(hyps))})
        if any(not block["global_assignment_agrees"] for block in continuity):
            failures.append("Speaker assignment changed across repeated sections")
    if baseline:
        for value, previous, name in (
            (cp["cpwer"] if cp else 1, baseline["speaker_attributed"]["cpwer"], "cpWER"),
            (metrics["diarization"]["boundary_250ms"]["der"], baseline["diarization"]["boundary_250ms"]["der"], "collared DER"),
        ):
            if value - previous > 0.05:
                failures.append(f"{name} regressed more than 5 percentage points")
    timing = reference_timing(clip, response)
    if any(not value["matched_passages"] for value in timing.values()):
        failures.append("Missing start/middle/end reference timestamp checks")
    return {"accepted": not failures, "failures": failures, "reference_timing": timing, "speaker_continuity": continuity}
