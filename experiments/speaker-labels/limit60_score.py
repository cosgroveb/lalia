"""Check native finish, matched tails and speaker continuity for the fixed hour trial."""

import math

from prepare_public import crop
from score import by_speaker, cp_word_error_rate, score
from limit60_runtime import DENOMINATOR_SHA, PROMPT_SHA


def one_audit(rows, key):
    matches = [row for row in rows if key in row]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key} audit")
    return matches[0]


def native_finish(rows, response):
    loaded = one_audit(rows, "loaded_layers")
    request = one_audit(rows, "model_max_tokens")
    builder = one_audit(rows, "builder_prompt_sha256")
    generator = one_audit(rows, "generator_max_tokens")
    terminal = one_audit(rows, "last_token_id")
    usage = response["usage"]
    if (loaded != {"loaded_layers": 28, "denominator_sha256": DENOMINATOR_SHA,
                   "weights_dtype": "bfloat16"}
            or request["prompt_sha256"] != PROMPT_SHA or request["model_max_tokens"] != 49152
            or 151645 not in request["eos_ids"] or builder["builder_prompt_sha256"] != PROMPT_SHA
            or generator["generator_max_tokens"] != 49152 or terminal["last_token_id"] != 151645
            or usage["type"] != "tokens" or not 0 < usage["output_tokens"] < 49152
            or terminal["yielded_tokens"] != usage["output_tokens"] + 1
            or builder["input_tokens"] != generator["generator_input_tokens"]
            or usage["input_tokens"] != builder["input_tokens"]
            or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
            or usage["input_tokens"] + 49152 > 131072):
        raise ValueError("Native finish, constructor or token audit disagrees with the response")
    return {"last_token_id": terminal["last_token_id"], "yielded_tokens": terminal["yielded_tokens"],
            "usage": usage, "max_tokens": 49152}


def tail_metrics(clip, response):
    end, start = clip["duration"], clip["duration"] - 300
    if start < 0:
        raise ValueError("Tail control must contain 300 seconds")
    tail = {"id": clip["id"] + "-tail", "duration": 300,
            "reference_segments": crop(clip["reference_segments"], start, end),
            "reference_turns": crop(clip["reference_turns"], start, end)}
    segments = response["segments"]
    metrics = score(tail, {"segments": crop(segments, start, end)}, expect_speakers=True)
    return {"metrics": metrics, "boundary_hypothesis_segments": sum(
        segment["start"] < start < segment["end"] or segment["start"] < end < segment["end"]
        for segment in segments)}


def quarter_continuity(clip, response):
    def assignment(reference, hypothesis):
        return dict(cp_word_error_rate(by_speaker(reference), by_speaker(hypothesis),
                                      reference_sort=False, hypothesis_sort=False).assignment)
    reference, hypothesis = clip["reference_segments"], response["segments"]
    global_mapping = assignment(reference, hypothesis)
    quarters = []
    for start in range(0, 3600, 900):
        refs, hyps = crop(reference, start, start + 900), crop(hypothesis, start, start + 900)
        mapping = assignment(refs, hyps)
        agrees = all(speaker in global_mapping and global_mapping[speaker] == label
                     for speaker, label in mapping.items())
        quarters.append({"start": start, "end": start + 900, "assignment": mapping,
                         "global_assignment_agrees": agrees})
    return {"global_assignment": global_mapping, "quarters": quarters}


def compare_tail(current, baseline):
    values = (
        ("cpWER", current["speaker_attributed"]["cpwer"], baseline["speaker_attributed"]["cpwer"], 1),
        ("collared DER", current["diarization"]["boundary_250ms"]["der"],
         baseline["diarization"]["boundary_250ms"]["der"], 1),
        ("speech coverage", current["diarization"]["speech_coverage"]["covered_reference_fraction"],
         baseline["diarization"]["speech_coverage"]["covered_reference_fraction"], -1),
    )
    deltas, failures = {}, []
    for name, actual, previous, direction in values:
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (actual, previous)):
            raise ValueError(f"Missing or nonfinite tail {name}")
        deltas[name] = actual - previous
        if direction * deltas[name] > 0.05 + 1e-12:
            failures.append(f"Tail {name} regressed more than five percentage points")
    return {"deltas": deltas, "failures": failures}
