import hashlib
import os
from pathlib import Path
import sys
import types
import unittest

import mlx.core as mx

import mlx.nn as nn
from mlx.utils import tree_flatten

import residual_silu as probe

RUNTIME = Path.home() / ".local/state/lalia-speaker-qualification/official-api-path/runtime"


def expected_report():
    return {"elements": 1537, "domain": [-16, 16], "mlx": "0.32.4", "torch": "2.14.1",
            "device": "Device(gpu, 0)", "identity_checks": dict.fromkeys(probe.IDENTITY_KEYS, True),
            "dtypes": dict.fromkeys(probe.DTYPE_KEYS, "mlx.core.bfloat16"),
            "historical_vs_explicit_differences": 0, "cpu_reference_fp32_sha256": probe.CANDIDATE_SHA,
            "comparisons": {name: {"different_from_cpu_torch": count, "max_abs_error": error, "fp32_sha256": digest}
                            for name, count, error, digest in (("historical", 458, 0.125, probe.NATIVE_SHA),
                                                              ("explicit", 458, 0.125, probe.NATIVE_SHA),
                                                              ("candidate", 0, 0.0, probe.CANDIDATE_SHA))}}


class FixedProjection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = value

    def __call__(self, _):
        return self.weight


class IdentityProjection(FixedProjection):
    def __call__(self, value):
        return value


def original_mlp(module):
    mlp = module.TextMLP.__new__(module.TextMLP)
    nn.Module.__init__(mlp)
    mlp.hidden_size, mlp.intermediate_size = 1024, 3072
    mlp.gate_proj = FixedProjection(mx.array([-1, 0, 1], dtype=mx.bfloat16))
    mlp.up_proj = FixedProjection(mx.array([1, 2, 3], dtype=mx.bfloat16))
    mlp.down_proj = IdentityProjection(mx.array([1], dtype=mx.bfloat16))
    return mlp


def model_stub(module):
    language = module.TextModel.__new__(module.TextModel)
    nn.Module.__init__(language)
    language.layers = []
    for _ in range(28):
        layer = nn.Module()
        layer.mlp = original_mlp(module)
        layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm = nn.Module(), nn.Module(), nn.Module()
        language.layers.append(layer)
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = language
    return model


class ResidualSiLUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            raise ValueError("Run CPU checks with nice -n 10")
        cls.addClassCleanup(mx.set_default_device, mx.default_device())
        mx.set_default_device(mx.cpu)
        cls.addClassCleanup(setattr, sys, "path", sys.path[:])
        sys.path.insert(0, str(RUNTIME))
        cls.module = probe.historical_text(RUNTIME)

    def test_selector_preserves_baseline_and_changes_only_candidate_instances(self):
        model, other = model_stub(self.module), model_stub(self.module)
        originals = [layer.mlp for layer in model.model.language_model.layers]
        outside = [(layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm)
                   for layer in model.model.language_model.layers]
        parameters = dict(tree_flatten(model.parameters()))
        original_call, original_silu = self.module.TextMLP.__call__, nn.silu
        report = probe.select_activation(model, "baseline", RUNTIME)
        self.assertEqual(report["replacement_count"], 0)
        self.assertTrue(all(layer.mlp is original for layer, original in zip(model.model.language_model.layers, originals)))
        report = probe.select_activation(model, "candidate", RUNTIME)
        self.assertEqual(report["replacement_count"], 28)
        for layer, original, preserved in zip(model.model.language_model.layers, originals, outside):
            self.assertIs(type(layer.mlp), probe.FullSiLUTextMLP)
            for actual, expected in zip((layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm), preserved):
                self.assertIs(actual, expected)
            for name in probe.PROJECTIONS:
                self.assertIs(getattr(layer.mlp, name), getattr(original, name))
                self.assertIs(getattr(layer.mlp, name).weight, getattr(original, name).weight)
        after = dict(tree_flatten(model.parameters()))
        self.assertEqual(after.keys(), parameters.keys())
        self.assertTrue(all(after[name] is value for name, value in parameters.items()))
        self.assertTrue(all(type(layer.mlp) is self.module.TextMLP for layer in other.model.language_model.layers))
        self.assertIs(self.module.TextMLP.__call__, original_call)
        self.assertIs(nn.silu, original_silu)

    def test_bad_final_layer_or_dtype_prevents_any_replacement(self):
        for bad in ("type", "dtype", "dimensions"):
            model = model_stub(self.module)
            originals = [layer.mlp for layer in model.model.language_model.layers]
            last = originals[-1]
            if bad == "type":
                model.model.language_model.layers[-1].mlp = types.SimpleNamespace()
            elif bad == "dtype":
                last.gate_proj.weight = mx.array([1], dtype=mx.float32)
            else:
                last.intermediate_size = 42
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.select_activation(model, "candidate", RUNTIME)
            self.assertTrue(all(layer.mlp is original for layer, original in zip(model.model.language_model.layers[:-1], originals[:-1])))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            probe.select_activation(None, "wide", RUNTIME)
        with self.assertRaisesRegex(ValueError, "pinned historical source"):
            probe.historical_text(Path("/untrusted-runtime"))

    def test_candidate_uses_compiled_helper_and_preserves_bf16_product(self):
        original = original_mlp(self.module)
        replacement = probe.FullSiLUTextMLP(original)
        self.assertIs(replacement.activation, probe.full_silu_bf16)
        result = replacement(mx.array([0], dtype=mx.bfloat16))
        expected = probe.full_silu_bf16(original.gate_proj.weight) * original.up_proj.weight
        mx.eval(result, expected)
        self.assertEqual(result.dtype, mx.bfloat16)
        self.assertTrue(mx.array_equal(result, expected).item())
        for projection in (original.gate_proj, original.up_proj):
            previous = projection.weight
            projection.weight = previous.astype(mx.float32)
            with self.assertRaisesRegex(ValueError, "BF16"):
                replacement(None)
            projection.weight = previous

    def test_preflight_rejects_each_changed_or_missing_observation(self):
        probe.require_preflight(expected_report())
        for group in ("identity_checks", "dtypes", "comparisons"):
            for key in expected_report()[group]:
                changed = expected_report()
                del changed[group][key]
                with self.subTest(group=group, missing=key), self.assertRaises((ValueError, KeyError)):
                    probe.require_preflight(changed)
        for name in ("historical", "explicit", "candidate"):
            for field, bad in (("different_from_cpu_torch", 1), ("max_abs_error", 0.01), ("fp32_sha256", "wrong")):
                changed = expected_report()
                changed["comparisons"][name][field] = bad
                with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                    probe.require_preflight(changed)
        for field, bad in (("elements", 1536), ("domain", [-15, 15]), ("device", "Device(cpu, 0)"),
                           ("mlx", "0.32.5"), ("torch", "2.14.2"), ("historical_vs_explicit_differences", 1),
                           ("cpu_reference_fp32_sha256", "wrong")):
            changed = expected_report()
            changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.require_preflight(changed)

    def test_fixed_cpu_reference_reproduces_saved_candidate_hash(self):
        import numpy as np
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        gate, up, reference = probe.fixed_cpu_reference()
        self.assertEqual((gate.size, float(gate[0]), float(gate[-1])), (1537, -16, 16))
        self.assertEqual((gate.dtype, up.dtype, reference.dtype), (np.float32, np.float32, np.float32))
        self.assertEqual(hashlib.sha256(reference.tobytes()).hexdigest(), probe.CANDIDATE_SHA)
        self.assertEqual(probe.compare_cpu(reference, reference)["different_from_cpu_torch"], 0)

    def test_gpu_preflight_cannot_run_as_cpu_unit_check(self):
        with self.assertRaisesRegex(ValueError, "reviewed Metal worker"):
            probe.operator_preflight(Path("/unused"), RUNTIME)


if __name__ == "__main__":
    unittest.main()
