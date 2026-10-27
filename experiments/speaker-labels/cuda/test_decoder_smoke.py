from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

import decoder_smoke as smoke


class DecoderSmokeTests(unittest.TestCase):
    def control_metadata(self):
        return {"operators": {
            "cpu_math": ["aten::_scaled_dot_product_attention_math"],
            "cuda_math": ["aten::_scaled_dot_product_attention_math"],
            "cuda_fused": ["aten::_scaled_dot_product_flash_attention"]},
            "cache_lengths": {"cpu_math": 65, "cuda_math": 65, "cuda_fused": 65},
            "available_bytes_after": 8 * 1024**3}

    def test_failed_control_preserves_logits_hashes_metrics_and_original_failure(self):
        reference = torch.arange(128).bfloat16() / 16
        logits = {name: reference.clone() for name in ("cpu_math", "cuda_fused", "cuda_math")}
        logits["cuda_fused"][0] += 0.25
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(smoke, "require_outer_product", return_value="_native::bmm_triton_0"):
                with self.assertRaisesRegex(ValueError, "did not qualify"):
                    smoke.finish_control(root, logits, self.control_metadata())
            saved = load_file(root / "decoder-logits.safetensors")
            result = json.loads((root / "decoder-control.json").read_text())
            for name, values in logits.items():
                self.assertTrue(torch.equal(values, saved[name]))
                self.assertEqual(result["logit_hashes"][name], smoke.probe.tensor_sha256(values))
            self.assertEqual(set(result["pairwise"]), {"fused_vs_cpu", "math_vs_cpu", "fused_vs_math"})
            self.assertEqual(result["original_fused_parity_gate"], "FAILED")
            self.assertFalse(result["control_qualified"])
            self.assertTrue(result["cuda_math_matches_cpu"])

    def test_nonfinite_logits_are_saved_before_validation_fails(self):
        logits = {name: torch.ones(128, dtype=torch.bfloat16) for name in ("cpu_math", "cuda_fused", "cuda_math")}
        logits["cuda_math"][0] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                smoke.finish_control(root, logits, self.control_metadata())
            saved = load_file(root / "decoder-logits.safetensors")
            self.assertTrue(torch.isnan(saved["cuda_math"][0]))
            self.assertTrue((root / "decoder-control.json").is_file())

    def test_control_requires_both_math_agreement_and_reproduced_fused_failure(self):
        observed = {"outside_tolerance": 2, "max_abs": 0.0078125, "rms": 0.0027748744469136}
        passed = {"outside_tolerance": 0, "max_abs": 0, "rms": 0}
        logits = {name: torch.ones(128, dtype=torch.bfloat16) for name in ("cpu_math", "cuda_fused", "cuda_math")}
        for fused, math, qualifies in ((observed, passed, True), (passed, passed, False), (observed, observed, False)):
            with self.subTest(qualifies=qualifies), tempfile.TemporaryDirectory() as temporary:
                with patch.object(smoke, "metrics", side_effect=[fused, math, observed]), \
                        patch.object(smoke, "require_outer_product", return_value="_native::bmm_triton_0"):
                    if qualifies:
                        result = smoke.finish_control(Path(temporary), logits, self.control_metadata())
                        self.assertTrue(result["control_qualified"])
                        self.assertEqual(result["original_fused_parity_gate"], "FAILED")
                    else:
                        with self.assertRaisesRegex(ValueError, "did not qualify"):
                            smoke.finish_control(Path(temporary), logits, self.control_metadata())

    def test_outer_product_requires_registered_function_identities_and_exact_profile_name(self):
        from torch._native import registry
        from torch._native.ops.bmm_outer_product import triton_impl
        node = SimpleNamespace(active=True, dsl_name="triton", node_id="bmm_triton_0",
                               impl_fn=triton_impl._bmm_outer_product_impl,
                               cond_fn=triton_impl._bmm_outer_product_cond)
        with patch.dict(registry._graphs, {("bmm", "CUDA"): [node]}, clear=True):
            self.assertEqual(smoke.require_outer_product(["_native::bmm_triton_0"]),
                             "_native::bmm_triton_0")
            for names in (["aten::bmm"], ["_native::bmm_triton_1"], []):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    smoke.require_outer_product(names)
            for field in ("impl_fn", "cond_fn"):
                with patch.object(node, field, lambda: None):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        smoke.require_outer_product(["_native::bmm_triton_0"])

    def test_complete_cpu_decoder_uses_production_heads_and_fresh_caches(self):
        torch.set_num_threads(2)
        model, embeddings = smoke.fixture()
        self.assertEqual((model.config.num_attention_heads, model.config.num_key_value_heads,
                          model.config.head_dim), (16, 8, 128))
        self.assertEqual(model.rotary_emb.inv_freq.dtype, torch.bfloat16)
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            first, length = smoke.score(model, embeddings)
            second, repeated = smoke.score(model, embeddings)
            direct = model(inputs_embeds=embeddings[None], use_cache=False,
                           position_ids=torch.arange(smoke.OFFSET, smoke.OFFSET + smoke.LENGTH)[None])
            direct_logits = torch.nn.functional.linear(direct.last_hidden_state[:, -1],
                                                       model.embed_tokens.weight)[0]
        self.assertEqual((length, repeated), (65, 65))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(smoke.compare(first, second)["outside_tolerance"], 0)
        self.assertEqual(smoke.compare(first, direct_logits)["outside_tolerance"], 0)
        self.assertEqual(smoke.OFFSET + smoke.LENGTH - 1, 88813)
        for value in (0, 0.25, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke.compare(torch.full_like(first, value), first)

    def test_zero_exit_without_result_and_fatal_output_fail(self):
        for output in ("finished\n", "FATAL: kernel mismatch\n"):
            with tempfile.TemporaryDirectory() as temporary:
                log = Path(temporary) / "worker.log"
                def execute(*args, **kwargs):
                    kwargs["stdout"].write(output)
                    return type("Completed", (), {"returncode": 0})()
                with patch.object(smoke.subprocess, "run", side_effect=execute):
                    with self.assertRaises((RuntimeError, ValueError)):
                        smoke.run(log)


if __name__ == "__main__":
    unittest.main()
