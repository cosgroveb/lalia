"""Check patched model/engine diagnostic hooks without loading model weights.

Run with the private runtime's Python and PYTHONPATH pointing at patched
package copies. Only audio decoding and generated tokens are fixtures.
"""

import asyncio
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from mlx_audio.stt.models.moss_transcribe_diarize import Model
from omlx.engine.stt import STTEngine, TranscriptionControl


class Tokenizer:
    fail = False

    def decode(self, tokens, **kwargs):
        if self.fail:
            raise ValueError("tokenizer failed")
        return "".join({10: "[0.00]", 20: "[S01]"}[token] for token in tokens)


class FixtureModel:
    sample_rate = 16000

    def __init__(self, reason, guard):
        self.reason = reason
        self.guard = guard
        self._tokenizer = Tokenizer()
        self._tokenizer.fail = reason == "tokenizer"

    def _prepare_generation_inputs(self, *args):
        return None, None, 4, 2

    def _eos_token_ids(self):
        return {99}

    def _parse_segments(self, *args):
        if self.reason == "segments":
            raise ValueError("segments failed")
        if self.reason == "late_cancel":
            self.guard.cancelled.set()
        return []

    def generate(self, audio, **kwargs):
        return Model.generate(self, audio, **kwargs)


class HookTests(unittest.TestCase):
    def test_runtime_preserves_failures_and_leaves_success_alone(self):
        messages = {"token_limit": "token budget", "timeout": "time limit", "cancelled": "cancelled",
                    "error": "decoder failed", "tokenizer": "tokenizer failed", "segments": "segments failed",
                    "late_cancel": "cancelled"}
        for reason in (*messages, "eos"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / "audio.wav"
                audio.write_bytes(b"fixture")
                guard = TranscriptionControl(2, max_tokens=2)
                engine = STTEngine("MOSS")
                engine._model = FixtureModel(reason, guard)

                def tokens(**kwargs):
                    yield 10, None
                    if reason == "timeout":
                        guard.deadline = 0
                        kwargs["stop_check"]()
                    elif reason == "cancelled":
                        guard.cancelled.set()
                        kwargs["stop_check"]()
                    elif reason == "error":
                        raise RuntimeError("decoder failed")
                    yield (20 if reason == "token_limit" else 99), None

                with patch("mlx_audio.lm.generate.generate_step", tokens), \
                     patch("mlx_audio.audio_io.read", return_value=(np.zeros(32000), 16000)), \
                     patch("pathlib.Path.home", return_value=root):
                    if reason == "eos":
                        result = asyncio.run(engine.transcribe(audio, _lalia_control=guard))
                        self.assertEqual(result["text"], "[0.00]")
                    else:
                        with self.assertRaisesRegex((RuntimeError, TimeoutError, ValueError), messages[reason]):
                            asyncio.run(engine.transcribe(audio, _lalia_control=guard))
                saved = list(root.rglob("*.json"))
                self.assertEqual(len(saved), 0 if reason == "eos" else 1)
                if saved:
                    diagnostic = json.loads(saved[0].read_text())
                    expected_reason = {"tokenizer": "error", "segments": "error", "late_cancel": "cancelled"}.get(reason, reason)
                    self.assertEqual(diagnostic["stop_reason"], expected_reason)
                    self.assertEqual(diagnostic["token_ids"], [10, 20] if reason == "token_limit" else [10])
                    expected_text = None if reason == "tokenizer" else "[0.00][S01]" if reason == "token_limit" else "[0.00]"
                    self.assertEqual(diagnostic["text"], expected_text)


if __name__ == "__main__":
    unittest.main()
