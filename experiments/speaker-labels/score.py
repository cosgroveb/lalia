#!/usr/bin/env python3
"""Score local oMLX responses without emitting transcript text.

Requires jiwer==4.0.0, meeteval==0.4.3, pyannote.metrics==4.0.0,
typing_extensions==4.15.0 (undeclared dependency of pyannote.metrics).
WER serializes turns by start time and is order-sensitive during overlap.
cpWER optimally matches complete speaker transcripts. DER includes overlap;
its UEM is the whole clip, with both zero and +/-250 ms boundary collars.
"""

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import re
import unicodedata

import jiwer
from meeteval.wer.wer.cp import cp_word_error_rate
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


NON_SPEECH_EVENTS = frozenset({"[Environmental Sounds]", "[Human Sounds]", "[Silence]", "[Music]"})


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = text.replace("'", "")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def clean_text(text):
    return re.sub(r"^\s*\[S\d+\]\s*", "", text)


def has_speaker(segment):
    value = segment.get("speaker_id")
    return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() != ""


def valid_time(segment, duration):
    start, end = segment.get("start"), segment.get("end")
    return (
        isinstance(start, (int, float)) and not isinstance(start, bool)
        and isinstance(end, (int, float)) and not isinstance(end, bool)
        and math.isfinite(start) and math.isfinite(end)
        and 0 <= start < end <= duration + 0.5
    )


def ordered(segments):
    return sorted(segments, key=lambda s: s.get("start", 0) if isinstance(s.get("start", 0), (int, float)) else 0)


def transcript(segments):
    return normalize(" ".join(clean_text(s.get("text", "")) for s in ordered(segments)))


def by_speaker(segments):
    groups = defaultdict(list)
    for segment in ordered(segments):
        groups[str(segment["speaker_id"])].append(clean_text(segment.get("text", "")))
    return {speaker: normalize(" ".join(text)) for speaker, text in groups.items()}


def annotation(segments):
    result = Annotation(uri="clip")
    for i, segment in enumerate(segments):
        result[Segment(segment["start"], segment["end"]), i] = str(segment["speaker_id"])
    return result


def lexical(reference_text, hypothesis_text):
    result = jiwer.process_words(reference_text, hypothesis_text)
    reference_words = len(reference_text.split())
    return {
        "wer": result.wer if reference_words else None,
        "reference_words": reference_words,
        "hypothesis_words": len(hypothesis_text.split()),
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "errors": result.substitutions + result.deletions + result.insertions,
    }


def score(clip, response, expect_speakers=False):
    """Return aggregate metrics; never include reference or hypothesis text."""
    duration = float(clip["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Clip duration must be finite and positive")
    reference = clip.get("reference_segments")
    raw_segments = response.get("segments")
    if raw_segments is None:
        raw_segments = []
    if not isinstance(raw_segments, list):
        raise ValueError("Response segments must be a list")
    valid_segments = [s for s in raw_segments if isinstance(s, dict) and isinstance(s.get("text"), str)]
    # Exact event tags describe non-speech; reference speech remains fully scored.
    events = Counter(s["text"] for s in valid_segments if s["text"] in NON_SPEECH_EVENTS)
    segments = [s for s in valid_segments if s["text"] not in NON_SPEECH_EVENTS]
    timed = [s for s in segments if valid_time(s, duration)]
    labeled = [s for s in segments if has_speaker(s)]
    raw_text = response.get("text", "")
    if not isinstance(raw_text, str):
        raise ValueError("Response text must be a string")
    hypothesis_text = transcript(segments) if valid_segments else normalize(raw_text) if not expect_speakers else ""
    invalid_times = len(segments) - len(timed)
    missing_labels = len(segments) - len(labeled)
    parsed = bool(valid_segments) or not raw_text.strip()
    result = {
        "clip_id": clip.get("id"),
        "audio_seconds": duration,
        "normalization": "NFKC, lowercase, remove apostrophes, punctuation/symbols to spaces; no number expansion",
        "versions": {name: importlib.metadata.version(name) for name in ("jiwer", "meeteval", "pyannote.metrics")},
        "schema": {
            "segment_count": len(raw_segments),
            "speech_segment_count": len(segments),
            "non_speech_event_count": sum(events.values()),
            "non_speech_event_counts": dict(events),
            "invalid_segment_count": len(raw_segments) - len(valid_segments),
            "invalid_time_count": invalid_times,
            "missing_speaker_count": missing_labels,
            "speaker_count": len({str(s["speaker_id"]) for s in labeled}),
            "speaker_output_parsed": parsed if expect_speakers else None,
            "last_segment_end": max((s["end"] for s in valid_segments if valid_time(s, duration)), default=None),
            "last_speech_end": max((s["end"] for s in timed), default=None),
            "hypothesis_words": len(hypothesis_text.split()),
        },
        "lexical": None,
        "speaker_attributed": None,
        "diarization": None,
    }
    if reference is None:
        result["reference_available"] = False
        return result
    if any(not has_speaker(s) or not valid_time(s, duration) or not isinstance(s.get("text"), str) for s in reference):
        raise ValueError("Invalid reference word segment")
    result["reference_available"] = True
    result["lexical"] = lexical(transcript(reference), hypothesis_text)
    result["schema"]["reference_speaker_count"] = len({str(s["speaker_id"]) for s in reference})
    result["schema"]["reference_last_word_end"] = max((s["end"] for s in reference), default=None)
    if not expect_speakers:
        result["speaker_metric_unavailable_reason"] = "Transcription-only baseline"
        return result
    if missing_labels or len(raw_segments) != len(valid_segments):
        result["speaker_metric_unavailable_reason"] = "Malformed or unlabeled output segments"
        return result
    ref_groups, hyp_groups = by_speaker(reference), by_speaker(segments)
    if result["lexical"]["reference_words"]:
        cp = cp_word_error_rate(ref_groups, hyp_groups, reference_sort=False, hypothesis_sort=False)
        result["speaker_attributed"] = {
            "cpwer": float(cp.error_rate), "errors": int(cp.errors), "reference_words": int(cp.length),
            "insertions": int(cp.insertions), "deletions": int(cp.deletions), "substitutions": int(cp.substitutions),
            "missed_speakers": int(cp.missed_speaker), "extra_speakers": int(cp.falarm_speaker),
        }
    turns = clip.get("reference_turns", reference)
    if any(not has_speaker(s) or not valid_time(s, duration) for s in turns):
        raise ValueError("Invalid reference diarization turn")
    ref_annotation = annotation(turns)
    hyp_annotation = annotation([s for s in timed if has_speaker(s)])
    uem = Timeline([Segment(0, duration)], uri="clip")
    result["diarization"] = {}
    for boundary_tolerance in (0, 0.25):
        metric = DiarizationErrorRate(collar=2 * boundary_tolerance, skip_overlap=False)
        components = metric(ref_annotation, hyp_annotation, uem=uem, detailed=True)
        key = "strict" if boundary_tolerance == 0 else "boundary_250ms"
        result["diarization"][key] = {
            "der": float(components["diarization error rate"]) if components["total"] else None,
            "reference_speaker_seconds": float(components["total"]),
            "missed_speaker_seconds": float(components["missed detection"]),
            "false_alarm_speaker_seconds": float(components["false alarm"]),
            "confusion_speaker_seconds": float(components["confusion"]),
            "boundary_tolerance_seconds": boundary_tolerance,
            "overlap_included": True,
        }
    ref_timeline = ref_annotation.get_timeline().support().crop(uem)
    hyp_timeline = hyp_annotation.get_timeline().support().crop(uem)
    speech_seconds = ref_timeline.duration()
    result["diarization"]["speech_coverage"] = {
        "reference_union_seconds": speech_seconds,
        "hypothesis_union_seconds": hyp_timeline.duration(),
        "covered_reference_fraction": ref_timeline.crop(hyp_timeline).duration() / speech_seconds if speech_seconds else None,
        "invalid_time_segments_excluded": invalid_times,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--speaker-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = manifest["clips"] if isinstance(manifest, dict) else manifest
    matches = [clip for clip in clips if clip["id"] == args.clip]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching clip")
    result = score(matches[0], json.loads(args.response.read_text()), args.speaker_model)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
