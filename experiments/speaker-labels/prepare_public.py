#!/usr/bin/env python3
"""Prepare one AMI test meeting and a three-minute, four-speaker excerpt."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import wave
from xml.etree import ElementTree
from zipfile import ZipFile


MEETING = "ES2004a"
MIRROR_REVISION = "722d8891643e1e4dc62cfd0d198fa05a1646c3cc"
SETUP_REVISION = "67c2d539286e89f68952d5dcf83912bd9f01dfae"
MIRROR = f"https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/resolve/{MIRROR_REVISION}"
SOURCES = {
    "ES2004a.Mix-Headset.wav": (
        "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav",
        "3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f",
    ),
    "ami_public_manual_1.6.2.zip": (
        f"{MIRROR}/annotations/ami_public_manual_1.6.2.zip",
        "56e29112cd89cfba76fbf8f10215eb245cc89b0d3d27355ba14b406a7ea7776c",
    ),
    "ES2004a.rttm": (
        f"https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/{SETUP_REVISION}/only_words/rttms/test/ES2004a.rttm",
        "9869c6146c2fd9595403edb36c2caeda65c12ffa2c0af4ce48d6814b673fd5a9",
    ),
}


def fetch(path, url, expected_hash):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--max-time", "600", "--output", str(temporary), url],
            check=True,
        )
        temporary.replace(path)
    with path.open("rb") as source:
        actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"SHA256 mismatch for {path.name}: {actual_hash}")


def words_from_archive(path):
    words = []
    excluded = Counter()
    with ZipFile(path) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        meeting = next(m for m in meetings.iter("meeting") if m.get("observation") == MEETING)
        speaker_ids = {s.get("nxt_agent"): s.get("global_name") for s in meeting.findall("speaker")}
        for agent, speaker in speaker_ids.items():
            root = ElementTree.fromstring(archive.read(f"words/{MEETING}.{agent}.words.xml"))
            for element in root:
                if element.tag != "w" or element.get("punc") == "true":
                    excluded["punctuation_or_non_word_event"] += 1
                    continue
                start = float(element.attrib["starttime"])
                end = float(element.attrib["endtime"])
                if end <= start:
                    excluded["nonpositive_word_duration"] += 1
                    continue
                text = "".join(element.itertext()).strip()
                if not text:
                    excluded["empty_word"] += 1
                    continue
                words.append({"start": start, "end": end, "text": text, "speaker_id": speaker})
        (path.parent / "AMI-LICENCE.txt").write_bytes(archive.read("LICENCE.txt"))
    return sorted(words, key=lambda word: (word["start"], word["end"], word["speaker_id"])), dict(excluded)


def turns_from_rttm(path):
    turns = []
    for line in path.read_text().splitlines():
        fields = line.split()
        start = float(fields[3])
        turns.append({"start": start, "end": start + float(fields[4]), "speaker_id": fields[7]})
    return turns


def crop(segments, start, end):
    return [dict(segment, start=max(start, segment["start"]) - start,
                 end=min(end, segment["end"]) - start)
            for segment in segments if segment["end"] > start and segment["start"] < end]


def stress_clip(original, output, words, turns, minutes):
    """Repeat the meeting, ending with its tail so final speech stays known."""
    destination = output / f"ami-{MEETING}-stress-{minutes}m.wav"
    duration = minutes * 60
    reference_words, reference_turns, blocks = [], [], []
    with wave.open(str(original)) as source, wave.open(str(destination), "wb") as target:
        target.setparams(source.getparams())
        rate, frames = source.getframerate(), source.getnframes()
        written = 0
        while written < duration * rate:
            count = min(frames, duration * rate - written)
            start = frames - count if count < frames else 0
            source.setpos(start)
            offset = written / rate
            for reference, entries in ((reference_words, words), (reference_turns, turns)):
                reference.extend(dict(item, start=item["start"] + offset, end=item["end"] + offset)
                                 for item in crop(entries, start / rate, (start + count) / rate))
            blocks.append({"start": offset, "end": offset + count / rate, "source_start": start / rate})
            remaining = count
            while remaining:
                chunk = min(remaining, rate * 30)
                target.writeframes(source.readframes(chunk))
                remaining -= chunk
            written += count
    return {"id": destination.stem, "audio_path": str(destination), "duration": duration,
            "private": False, "stress_fixture": True, "repeated_blocks": blocks,
            "reference_segments": reference_words, "reference_turns": reference_turns,
            "words_per_speaker": dict(Counter(word["speaker_id"] for word in reference_words))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stress-minutes", type=int, nargs="*", choices=(60, 90), default=[])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in SOURCES.items():
        fetch(output / name, url, checksum)
    words, excluded = words_from_archive(output / "ami_public_manual_1.6.2.zip")
    turns = turns_from_rttm(output / "ES2004a.rttm")
    original = output / "ES2004a.Mix-Headset.wav"
    with wave.open(str(original)) as audio:
        full_duration = audio.getnframes() / audio.getframerate()
        assert audio.getnchannels() == 1 and audio.getframerate() == 16000
    clips = []
    for suffix, start, end in [("short", 720.35, 900.07), ("full", 0.0, full_duration)]:
        destination = output / f"ami-{MEETING}-{suffix}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(original),
             "-ss", str(start), "-t", str(end - start), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(destination)],
            check=True,
        )
        with wave.open(str(destination)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        reference_words = crop(words, start, end)
        reference_turns = crop(turns, start, end)
        assert len({word["speaker_id"] for word in reference_words}) == 4
        assert all(0 <= item["start"] < item["end"] <= duration + 0.001
                   for item in reference_words + reference_turns)
        boundary_words = sum(word["start"] < start < word["end"] or word["start"] < end < word["end"]
                             for word in words)
        clips.append({
            "id": f"ami-{MEETING}-{suffix}", "audio_path": str(destination),
            "duration": duration, "source_start": start, "source_end": end, "private": False,
            "reference_segments": reference_words, "reference_turns": reference_turns,
            "boundary_words_clipped": boundary_words,
            "words_per_speaker": dict(Counter(word["speaker_id"] for word in reference_words)),
        })
    clips.extend(stress_clip(original, output, words, turns, minutes) for minutes in args.stress_minutes)
    manifest = {
        "dataset": "AMI Meeting Corpus, ES2004a, headset mix, official full-corpus ASR test split",
        "license": "CC-BY-4.0",
        "attribution": "AMI Consortium / University of Edinburgh; Carletta et al., The AMI Meeting Corpus: A Pre-announcement (2005)",
        "sources": {name: {"url": url, "sha256": checksum} for name, (url, checksum) in SOURCES.items()},
        "reference_policy": "Manual v1.6.2 words; exclude punctuation, non-word events and nonpositive-duration words; retain lexical truncations. DER uses pyannote only_words RTTM. Short excerpt selected for all four speakers, before inference.",
        "excluded_annotation_entries": excluded,
        "caveats": ["One meeting cannot establish general accuracy.",
                    "Close-talking headset mix is cleaner than ambient MacBook audio.",
                    "Overlapping speech is retained; chronological WER depends on tie ordering.",
                    "Model training overlap with this public benchmark is unknown.",
                    "The XML meeting metadata duration is stale; duration comes from audio sample count."],
        "clips": clips,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "excluded": excluded, "clips": [
        {k: clip[k] for k in ("id", "duration", "words_per_speaker", "stress_fixture") if k in clip}
        for clip in clips]}, indent=2))


if __name__ == "__main__":
    main()
