import json
from pathlib import Path
import tempfile
import unittest

import torch
import smoke_numeric as smoke


class NumericSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_attention_reference_matches_cpu_math_sdpa(self):
        q, k, v = smoke.fixture(16, 64, torch.Generator().manual_seed(17))
        expected = smoke.reference_attention(q, k, v, [0, 8, 15], dtype=torch.float64)
        mask = torch.arange(64)[None] <= (torch.arange(16)[:, None] + 48)
        actual = torch.nn.functional.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                                                attn_mask=mask, enable_gqa=True)
        torch.testing.assert_close(actual[:, :, [0, 8, 15]], expected, atol=1e-12, rtol=1e-12)
        smoke.compare(expected.bfloat16(), expected)
        self.assertFalse(q.is_contiguous())

    def test_cache_layout_matches_initial_and_later_updates(self):
        for q_length, kv_length in ((16, 16), (16, 64), (1, 64)):
            q, k, v = smoke.fixture(q_length, kv_length, torch.Generator().manual_seed(17))
            self.assertEqual(q.stride(), (q_length * 16 * 128, 128, 16 * 128, 1))
            self.assertEqual(k.stride(), (8 * kv_length * 128, kv_length * 128, 128, 1))
            self.assertEqual(v.stride(), k.stride())

    def test_zero_and_finite_garbage_fail_oracle(self):
        q, k, v = smoke.fixture(16, 64, torch.Generator().manual_seed(17))
        expected = smoke.reference_attention(q, k, v, [0, 8, 15], dtype=torch.float64)
        for value in (0.0, 0.25, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke.compare(torch.full_like(expected, value), expected)

    def test_masked_fixture_rejects_unmasked_and_wrong_alignment(self):
        q, k, v = smoke.fixture(16, 64, torch.Generator().manual_seed(17))
        rows = [0, 8, 15]
        expected = smoke.reference_attention(q, k, v, rows, dtype=torch.float64)
        wrong_mask = torch.arange(64)[None] <= torch.arange(16)[:, None]
        for mask in (None, wrong_mask):
            actual = torch.nn.functional.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                                                    attn_mask=mask, enable_gqa=True)
            with self.assertRaises(ValueError):
                smoke.compare(actual[:, :, rows], expected)

    def test_gemm_reference_accepts_bf16_rounding(self):
        generator = torch.Generator().manual_seed(17)
        inputs = torch.randn(16, 128, generator=generator).bfloat16()
        weight = torch.randn(128, 128, generator=generator).bfloat16()
        expected = inputs.double() @ weight.double().T
        smoke.compare(expected.bfloat16(), expected)
        with self.assertRaises(ValueError):
            smoke.compare(torch.zeros_like(expected), expected)

    def test_fatal_diagnostic_rejects_zero_exit_and_success_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            log.write_text('FATAL: kernel mismatch\n' + json.dumps({"type":"numeric-result","checks":[]}) + '\n')
            with self.assertRaisesRegex(RuntimeError, "Fatal runtime diagnostic"):
                smoke.read_result(log, "tiny", 0)

    def test_zero_exit_without_measurements_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            log.write_text("worker completed\n")
            with self.assertRaisesRegex(ValueError, "Missing"):
                smoke.read_result(log, "tiny", 0)


if __name__ == "__main__":
    unittest.main()
