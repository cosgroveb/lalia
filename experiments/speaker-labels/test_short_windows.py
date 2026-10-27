from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

import short_windows as sw


def segment(start, end, speaker="S01", text="sample"):
    return {"start": start, "end": end, "speaker_id": speaker, "text": text}


class ShortWindowsTests(unittest.TestCase):
    def test_mapping_rejects_collision_and_unknown(self):
        centroids = {"S01": np.array([1., 0., 0.]), "S02": np.array([0., 1., 0.])}
        mapped = sw.match_groups({"local2": centroids["S01"], "local1": centroids["S02"]}, centroids)
        self.assertTrue(mapped["accepted"])
        self.assertEqual(mapped["mapping"], {"local1": "S02", "local2": "S01"})
        collision = sw.match_groups({"a": centroids["S01"], "b": centroids["S01"]}, centroids)
        self.assertFalse(collision["accepted"])
        self.assertEqual(collision["collisions"], {"S01": ["a", "b"]})
        rejected = sw.match_groups({"unknown": np.array([0., 0., 1.])}, centroids)
        self.assertFalse(rejected["accepted"])
        self.assertTrue(rejected["assignments"][0]["rejected"])
        self.assertEqual(rejected["assignments"][0]["scores"], {"S01": 0., "S02": 0.})

    def test_predicted_selection_excludes_overlap_and_uses_longest_spans(self):
        clips = sw.embedding_clips([segment(0, 20), segment(5, 10, "S02"), segment(22, 27, "S02"), segment(30, 34)])
        self.assertEqual(clips["S01"], [{"start_frame": 162400, "end_frame": 210400},
                                       {"start_frame": 2400, "end_frame": 50400}])
        self.assertEqual(clips["S02"][0]["start_frame"], 354400)

    def test_insufficient_clean_span_fails(self):
        with self.assertRaisesRegex(ValueError, "No clean 3s"):
            sw.embedding_clips([segment(0, 20), segment(5, 10, "S02")])

    def test_prefixed_events_are_excluded_from_enrollment(self):
        from qualification import completion
        from score import NON_SPEECH_EVENTS

        words = segment(0, 4, text="[S01] sample [Silence] words")
        for event in NON_SPEECH_EVENTS:
            with self.subTest(event=event):
                response = {"text": f"[0][S01]sample [Silence] words[4][5][S02]{event}[9]",
                            "segments": [words, segment(5, 9, "S02", f"[S02] {event}")],
                            "usage": {"type": "tokens", "input_tokens": 10,
                                      "output_tokens": 20, "total_tokens": 30}}
                completion(response, 9, sw.TOKENS)
                spoken = sw.speech(response["segments"])
                self.assertEqual(spoken, [words])
                self.assertEqual(set(sw.embedding_clips(spoken)), {"S01"})

    def test_shared_gap_avoids_crossing_and_refuses_no_gap(self):
        left = [segment(330, 359), segment(363, 390)]
        right = [segment(330, 361), segment(364, 390)]
        self.assertEqual(sw.shared_gap(left, right, 330, 390, 360)["cut"], 362)
        self.assertEqual(sw.shared_gap([segment(0, 1)], [segment(1.2, 2)], 0, 2, 1.1)["cut"], 1.1)
        with self.assertRaisesRegex(ValueError, "No shared"):
            sw.shared_gap([segment(330, 370)], [segment(360, 390)], 330, 390, 360)

    def test_join_preserves_offsets_text_events_and_full_accounting(self):
        left = sw.offset_segments([segment(350, 359, text="left"), segment(363, 370, text="redundant")], 0, 0, {"S01": "S02"})
        right = sw.offset_segments([segment(20, 29, text="redundant"), segment(33, 40, text="right"),
                                    segment(30, 34, text="[S01] [Silence]")], 330, 1, {"S01": "S02"})
        kept, audit = sw.join([left, right], [362], 1049)
        self.assertEqual([s["text"] for s in kept], ["left", "[S01] [Silence]", "right"])
        self.assertEqual(kept[-1]["start"], 363)
        self.assertEqual(kept[-1]["end"], 370)
        self.assertEqual(kept[-1]["source"], {"window": 1, "segment": 1, "local_speaker": "S01", "offset_seconds": 330})
        self.assertEqual(len(audit), 5)
        self.assertEqual(sum(row["retained"] for row in audit), 3)
        self.assertEqual(sum(row["event"] for row in audit), 1)
        with self.assertRaisesRegex(ValueError, "crosses ownership"):
            sw.join([sw.offset_segments([segment(359, 363)], 0, 0, {}), right], [362], 1049)

    def test_comparison_excludes_prefixed_events_without_changing_text(self):
        from score import score

        reference = [segment(0, 4, text="hello"), segment(8, 12, text="world")]
        clip = {"duration": 12, "reference_segments": reference}
        baseline = {"segments": reference}
        response = {"segments": [segment(0, 4, text="[S01] hello"),
                                 segment(4, 8, text="[S01] [Silence]"),
                                 segment(8, 12, text="[S01] world")]}
        original = [dict(s) for s in response["segments"]]
        result = sw.compare(clip, response, baseline,
                            {"metrics": score(clip, baseline, expect_speakers=True)}, [6])
        metrics = result["metrics"]
        self.assertEqual(metrics["schema"]["non_speech_event_counts"], {"[Silence]": 1})
        self.assertEqual(metrics["schema"]["hypothesis_words"], 2)
        self.assertEqual(metrics["speaker_attributed"]["cpwer"], 0)
        self.assertEqual(metrics["diarization"]["strict"]["der"], 0)
        self.assertEqual(result["seams"][0]["merged_insertions"], 0)
        self.assertEqual(response["segments"], original)

    def test_extraction_is_sample_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source.wav", Path(temporary) / "part.wav"
            pcm = np.arange(16000, dtype="<i2").tobytes()
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(pcm)
            result = sw.extract(source, output, 333, 9333)
            with wave.open(str(output)) as reader:
                self.assertEqual(reader.readframes(reader.getnframes()), pcm[666:18666])
            self.assertEqual(result["start_frame"], 333)
            self.assertEqual(result["end_frame"], 9333)
            self.assertEqual(result["duration"], 9000 / 16000)

    def test_reference_positions_distinguish_deletion_and_insertion(self):
        clip = {"reference_segments": [segment(i * 2 + 1, i * 2 + 2, text=word)
                                       for i, word in enumerate("one two three four five six".split())]}
        statuses, inserted, unplaced = sw.reference_alignment(clip, {"segments": [segment(1, 12, text="one extra two three four six")]})
        self.assertEqual(statuses[("S01", 4)][1], "delete")
        self.assertEqual(inserted, [3.5])
        self.assertEqual(unplaced, 0)

    def test_process_write_failure_still_reaps_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = sw.Guard(Path(temporary))
            with patch.object(sw, "save", side_effect=OSError("synthetic disk failure")):
                with self.assertRaisesRegex(OSError, "synthetic"):
                    try:
                        guard.launch([sys.executable, "-c", "import time; time.sleep(30)"], Path(temporary) / "log")
                    finally:
                        guard.close()
            self.assertIsNotNone(guard.children[0].poll())


if __name__ == "__main__":
    unittest.main()
