from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

import eos_rope as probe
import native_rope_basis as basis


class NativeRoPETests(unittest.TestCase):
    def test_fp32_denominators_match_rounded_inverse_frequencies(self):
        rope = probe.NativeBF16RoPE(probe.TOTAL)
        self.assertEqual(rope.frequencies["stored_sha256"], probe.BF16_FREQUENCY_SHA256)
        self.assertEqual(rope.freqs.dtype, mx.float32)
        with mx.stream(mx.cpu):
            inverse = 1 / (1000000.0 ** (mx.arange(0, 128, 2, dtype=mx.float32) / 128))
            expected = mx.reciprocal(inverse.astype(mx.bfloat16).astype(mx.float32))
            np.testing.assert_array_equal(np.array(rope.freqs), np.array(expected))
        self.assertEqual(rope.frequencies["cpu_reciprocal_mismatches"], 10)
        value = mx.zeros((1, 16, 1, 128), dtype=mx.bfloat16)
        with patch.object(mx.fast, "rope", return_value=value) as call:
            self.assertIs(rope(value, offset=88813), value)
            args, kwargs = call.call_args
            self.assertIs(args[0], value)
            self.assertEqual(args[1], 128)
            self.assertIs(kwargs.pop("freqs"), rope.freqs)
            self.assertEqual(kwargs, {"traditional": False, "base": None, "scale": 1.0, "offset": 88813})
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            rope(value, offset=probe.TOTAL)

    def test_reference_replacement_preserves_weights_and_other_models_even_on_failure(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mx.ones((2, 2))
                self.rope = nn.RoPE(128, traditional=False, base=1000000)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

        model = nn.Module()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        model.model.language_model.layers = [Layer(), Layer()]
        other = Attention()
        other_rope = other.rope
        originals = [layer.self_attn.rope for layer in model.model.language_model.layers]
        before = {name: np.array(value) for name, value in tree_flatten(model.parameters())}
        rope = probe.NativeBF16RoPE(probe.TOTAL)

        def fail(*args, **kwargs):
            self.assertTrue(all(layer.self_attn.rope is rope for layer in model.model.language_model.layers))
            current = dict(tree_flatten(model.parameters()))
            self.assertEqual(set(current), set(before))
            for name, value in current.items():
                np.testing.assert_array_equal(np.array(value), before[name])
            self.assertIs(other.rope, other_rope)
            raise RuntimeError("synthetic pass failure")

        with tempfile.TemporaryDirectory() as temporary, patch.object(probe, "token_pass", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic pass failure"):
                probe.reference_pass(model, None, Path(temporary), rope, began=0)
        self.assertEqual([layer.self_attn.rope for layer in model.model.language_model.layers], originals)
        self.assertIs(other.rope, other_rope)

    def test_basis_check_rejects_nonfinite_gross_error_or_wrong_denominator_blindness(self):
        row = {"finite": True, "max_abs_error": 0.001, "offset": 88813,
               "wrong_denominator_max_abs_error": 1.0}
        basis.check({"cases": [row]})
        for key, value in (("finite", False), ("max_abs_error", 0.1), ("wrong_denominator_max_abs_error", 0.0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                basis.check({"cases": [{**row, key: value}]})


if __name__ == "__main__":
    unittest.main()
