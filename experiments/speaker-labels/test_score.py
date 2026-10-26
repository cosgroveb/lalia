import json
import unittest

from score import score


def segment(start, end, speaker, text):
    return {"start": start, "end": end, "speaker_id": speaker, "text": text}


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.reference = [segment(1, 3, "Alice", "One two"), segment(4, 6, "Bob", "Three four")]
        self.clip = {"id": "synthetic", "duration": 10, "reference_segments": self.reference}

    def test_label_permutation_and_moss_prefix(self):
        hypothesis = [segment(1, 3, "S02", "[S02] One, two!"), segment(4, 6, "S01", "[S01] THREE FOUR.")]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["lexical"]["wer"], 0)
        self.assertEqual(result["speaker_attributed"]["cpwer"], 0)
        self.assertEqual(result["diarization"]["strict"]["der"], 0)

    def test_merged_speaker_penalized(self):
        hypothesis = [dict(s, speaker_id=0) for s in self.reference]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["lexical"]["wer"], 0)
        self.assertEqual(result["speaker_attributed"]["cpwer"], 1)
        self.assertEqual(result["diarization"]["strict"]["der"], 0.5)

    def test_empty_output_is_all_missed(self):
        result = score(self.clip, {"text": "", "segments": []}, True)
        self.assertEqual(result["lexical"]["wer"], 1)
        self.assertEqual(result["speaker_attributed"]["cpwer"], 1)
        self.assertEqual(result["diarization"]["strict"]["der"], 1)

    def test_missing_label_is_not_silently_ignored(self):
        hypothesis = [dict(s) for s in self.reference]
        del hypothesis[1]["speaker_id"]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["schema"]["missing_speaker_count"], 1)
        self.assertIsNone(result["speaker_attributed"])
        self.assertIsNone(result["diarization"])

    def test_overlap_is_included(self):
        self.clip["reference_segments"][1].update(start=2, end=4)
        result = score(self.clip, {"segments": self.reference[:1]}, True)
        self.assertEqual(result["diarization"]["strict"]["der"], 0.5)
        self.assertEqual(result["diarization"]["strict"]["missed_speaker_seconds"], 2)

    def test_collar_is_quarter_second_on_each_side(self):
        hypothesis = [dict(s, start=s["start"] + 0.2, end=s["end"] - 0.2) for s in self.reference]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertAlmostEqual(result["diarization"]["strict"]["der"], 0.2)
        self.assertEqual(result["diarization"]["boundary_250ms"]["der"], 0)

    def test_silent_reference_has_insertion_count(self):
        self.clip["reference_segments"] = []
        result = score(self.clip, {"text": "Hallucinated words"})
        self.assertIsNone(result["lexical"]["wer"])
        self.assertEqual(result["lexical"]["insertions"], 2)

    def test_silence_false_alarm_has_duration_but_no_der_denominator(self):
        self.clip["reference_segments"] = []
        result = score(self.clip, {"segments": self.reference}, True)
        self.assertIsNone(result["diarization"]["strict"]["der"])
        self.assertEqual(result["diarization"]["strict"]["false_alarm_speaker_seconds"], 4)

    def test_wrong_segment_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "segments must be a list"):
            score(self.clip, {"segments": {}}, True)

    def test_no_transcript_in_metrics(self):
        result = score(self.clip, {"text": "One two three four"})
        self.assertNotIn("One two", json.dumps(result))
        self.assertIsNone(result["speaker_attributed"])

    def test_invalid_timestamp_excluded_and_counted(self):
        hypothesis = [dict(s) for s in self.reference]
        hypothesis[1]["end"] = 400
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["schema"]["invalid_time_count"], 1)
        self.assertEqual(result["diarization"]["strict"]["der"], 0.5)

    def test_unparsed_structured_output_is_not_scored_as_words(self):
        result = score(self.clip, {"text": '[{"Content": "One two"', "segments": []}, True)
        self.assertFalse(result["schema"]["speaker_output_parsed"])
        self.assertEqual(result["lexical"]["hypothesis_words"], 0)

    def test_silence_event_is_not_a_hallucinated_word(self):
        clip = {"id": "silence20", "duration": 20, "reference_segments": []}
        response = {"text": '[{"Start":0,"End":20.0,"Content":"[Silence]"}]',
                    "segments": [{"start": 0, "end": 20.0, "text": "[Silence]"}]}
        original = json.dumps(response)
        result = score(clip, response, True)
        self.assertEqual(result["lexical"]["insertions"], 0)
        self.assertEqual(result["schema"]["hypothesis_words"], 0)
        self.assertEqual(result["schema"]["missing_speaker_count"], 0)
        self.assertEqual(result["schema"]["non_speech_event_counts"], {"[Silence]": 1})
        self.assertTrue(result["schema"]["speaker_output_parsed"])
        self.assertEqual(result["diarization"]["strict"]["false_alarm_speaker_seconds"], 0)
        self.assertEqual(json.dumps(response), original)

    def test_known_events_do_not_block_valid_speaker_metrics(self):
        events = ["[Environmental Sounds]", "[Human Sounds]", "[Silence]", "[Music]"]
        hypothesis = self.reference + [{"start": 0, "end": 10, "text": text} for text in events]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["lexical"]["wer"], 0)
        self.assertEqual(result["speaker_attributed"]["cpwer"], 0)
        self.assertEqual(result["diarization"]["strict"]["der"], 0)
        self.assertEqual(result["schema"]["non_speech_event_count"], 4)
        self.assertEqual(result["schema"]["segment_count"], 6)
        self.assertEqual(result["schema"]["speech_segment_count"], 2)

    def test_event_replacing_reference_speech_remains_missed(self):
        hypothesis = [self.reference[0], {"start": 4, "end": 6, "text": "[Silence]"}]
        result = score(self.clip, {"segments": hypothesis}, True)
        self.assertEqual(result["lexical"]["wer"], 0.5)
        self.assertEqual(result["lexical"]["deletions"], 2)
        self.assertEqual(result["speaker_attributed"]["cpwer"], 0.5)
        self.assertEqual(result["diarization"]["strict"]["der"], 0.5)
        self.assertEqual(result["diarization"]["strict"]["missed_speaker_seconds"], 2)

    def test_unknown_or_mixed_event_text_still_requires_speaker(self):
        for text in ("[Noise]", "[Silence] Three four", "Three four"):
            with self.subTest(text=text):
                hypothesis = [self.reference[0], {"start": 4, "end": 6, "text": text}]
                result = score(self.clip, {"segments": hypothesis}, True)
                self.assertEqual(result["schema"]["non_speech_event_count"], 0)
                self.assertEqual(result["schema"]["missing_speaker_count"], 1)
                self.assertIsNone(result["speaker_attributed"])
                self.assertIsNone(result["diarization"])


if __name__ == "__main__":
    unittest.main()
