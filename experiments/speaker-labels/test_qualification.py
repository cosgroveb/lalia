from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from qualification import acceptance, completion


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.response = {"text": "[0.00][S01]नमस्ते 🫠[1.00]",
                         "segments": [{"start": 0, "end": 1, "speaker_id": "S01", "text": "[S01] नमस्ते 🫠"}],
                         "usage": {"type": "tokens", "input_tokens": 12, "output_tokens": 5, "total_tokens": 17}}

    def test_complete_unicode_and_start_zero(self):
        self.assertTrue(completion(self.response, 10, 6)["eos"])

    def test_exhaustion_rejected(self):
        with self.assertRaisesRegex(ValueError, "exhausted"):
            completion(self.response, 10, 5)

    def test_discarded_raw_span_rejected(self):
        self.response["text"] += "unfinished speech"
        with self.assertRaisesRegex(ValueError, "Unmatched raw"):
            completion(self.response, 10, 6)

    def test_empty_eos_and_trailing_silence(self):
        self.response.update(text="", segments=[{"start": 0, "end": 10, "text": ""}])
        self.response["usage"].update(output_tokens=0, total_tokens=12)
        self.assertEqual(completion(self.response, 10, 6)["segments"], 0)

    def test_zero_output_tokens_reject_nonempty_raw_or_structured_speech(self):
        self.response["usage"].update(output_tokens=0, total_tokens=12)
        for text in (self.response["text"], ""):
            self.response["text"] = text
            with self.subTest(raw_text=bool(text)), self.assertRaisesRegex(ValueError, "zero output tokens"):
                completion(self.response, 10, 6)

    def test_timestamp_and_count_validation(self):
        self.response["segments"][0]["end"] = 11
        with self.assertRaisesRegex(ValueError, "timestamps"):
            completion(self.response, 10, 6)
        self.response["segments"][0]["end"] = 1
        self.response["usage"]["total_tokens"] = 18
        with self.assertRaisesRegex(ValueError, "usage"):
            completion(self.response, 10, 6)

    def test_plain_unassigned_fallback_requires_exact_coverage(self):
        self.response.update(text="Unassigned speech", segments=[{"start": 0, "end": 10, "text": "Unassigned speech"}])
        self.assertTrue(completion(self.response, 10, 6)["raw_coverage"])
        self.response["text"] += " [0.1][S01]incomplete"
        self.response["segments"][0]["text"] = self.response["text"]
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            completion(self.response, 10, 6)

    def test_overlap_allowed_and_usage_required(self):
        self.response["text"] += "[0.50][S02]Overlapping speech[1.50]"
        self.response["segments"].append({"start": 0.5, "end": 1.5, "speaker_id": "S02", "text": "[S02] Overlapping speech"})
        self.assertEqual(completion(self.response, 10, 6)["segments"], 2)
        del self.response["usage"]
        with self.assertRaisesRegex(ValueError, "Missing token"):
            completion(self.response, 10, 6)

    def test_long_qualification_requires_successful_baseline(self):
        metrics = {"schema": {"speaker_count": 4, "missing_speaker_count": 0},
                   "speaker_attributed": {"cpwer": 0.9524, "missed_speakers": 0, "extra_speakers": 0},
                   "diarization": {"boundary_250ms": {"der": 0.1}}}
        timing = {key: {"matched_passages": 1} for key in ("start", "middle", "end")}
        with patch("qualification.reference_timing", return_value=timing):
            for clip in ({"duration": 3600}, {"duration": 100, "stress_fixture": True},
                         {"duration": 100, "repeated_blocks": [{"start": 0, "end": 100}]}):
                with self.subTest(clip=clip), self.assertRaisesRegex(ValueError, "baseline"):
                    acceptance(clip, {}, metrics)
            self.assertTrue(acceptance({"duration": 1049}, {}, metrics)["accepted"])

            baseline = {"status": "complete", "model": "MOSS-Transcribe-Diarize", "audio_seconds": 1049,
                        "prompt": None,
                        "completion": {"eos": True, "raw_coverage": True},
                        "qualification": {"accepted": True, "failures": []}, "metrics": deepcopy(metrics)}
            baseline["metrics"]["speaker_attributed"]["cpwer"] = 0.1
            self.assertTrue(acceptance({"duration": 3600}, {}, baseline["metrics"], baseline)["accepted"])
            untracked = dict(baseline)
            del untracked["prompt"]
            with self.assertRaisesRegex(ValueError, "same prompt"):
                acceptance({"duration": 3600}, {}, baseline["metrics"], untracked)
            prompt = "请转写音频 🫠"
            prompted = dict(baseline, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            self.assertTrue(acceptance({"duration": 3600}, {}, baseline["metrics"], prompted, prompt)["accepted"])
            with self.assertRaisesRegex(ValueError, "same prompt"):
                acceptance({"duration": 3600}, {}, baseline["metrics"], prompted)
            self.assertIn("cpWER regressed more than 5 percentage points",
                          acceptance({"duration": 3600}, {}, metrics, baseline)["failures"])
            for invalid in (None, {}, [], {"status": "rejected"},
                            dict(baseline, status="failed"), dict(baseline, completion={}),
                            dict(baseline, qualification={"accepted": False, "failures": ["failed"]}),
                            dict(baseline, metrics={}), dict(baseline, audio_seconds=3600)):
                for duration in (1049, 3600):
                    if invalid is None and duration < 3600:
                        continue
                    with self.subTest(baseline=invalid, duration=duration), self.assertRaisesRegex(ValueError, "baseline"):
                        acceptance({"duration": duration}, {}, metrics, invalid)
            for value in (None, True, "0.1", -1, float("nan"), float("inf")):
                invalid = deepcopy(baseline)
                invalid["metrics"]["speaker_attributed"]["cpwer"] = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "baseline"):
                    acceptance({"duration": 3600}, {}, metrics, invalid)


if __name__ == "__main__":
    unittest.main()
