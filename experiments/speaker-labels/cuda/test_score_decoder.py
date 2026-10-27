import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

import score_decoder as probe


def config():
    return Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=8, max_position_embeddings=131072, rope_theta=1000000,
                       tie_word_embeddings=True, pad_token_id=0, attention_dropout=0.0)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chunks_score_position_88813_once(self):
        parts = probe.chunks()
        self.assertEqual(parts[-1], (88813, 88814))
        self.assertEqual(sum(b - a for a, b in parts), 88814)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(parts, parts[1:])))

    def test_final_logits_reject_nonfinite_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            logits = torch.zeros(151936, dtype=torch.bfloat16)
            logits[42] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Nonfinite final logits"):
                probe.summary(logits)

    def test_official_loader_cast_changes_rope_buffer(self):
        model = probe.empty_decoder(config())
        self.assertEqual(model.embed_tokens.weight.dtype, torch.bfloat16)
        self.assertEqual(model.rotary_emb.inv_freq.dtype, torch.float32)
        full = model.rotary_emb.inv_freq.clone()
        model.to(dtype=torch.bfloat16)
        self.assertEqual(model.rotary_emb.inv_freq.dtype, torch.bfloat16)
        self.assertFalse(torch.equal(full, model.rotary_emb.inv_freq.float()))

    def test_fp32_rope_restores_original_precision_without_changing_weights_or_scaling(self):
        c = config()
        c.head_dim = 128
        model = probe.Qwen3Model(c).eval()
        expected = model.rotary_emb.inv_freq.clone()
        model.to(dtype=torch.bfloat16)
        rounded = model.rotary_emb.inv_freq.clone()
        original_buffer = model.rotary_emb.original_inv_freq.clone()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertEqual(probe.tensor_sha256(rounded), probe.R7_ROPE_SHA256)
        probe.restore_fp32_rope(model)
        self.assertEqual(model.rotary_emb.inv_freq.dtype, torch.float32)
        self.assertEqual(model.rotary_emb.inv_freq.device.type, "cpu")
        self.assertTrue(torch.equal(model.rotary_emb.inv_freq, expected))
        self.assertFalse(torch.equal(model.rotary_emb.inv_freq, rounded.float()))
        self.assertTrue(torch.equal(model.rotary_emb.inv_freq.bfloat16(), rounded))
        self.assertTrue(torch.equal(model.rotary_emb.original_inv_freq, original_buffer))
        self.assertEqual(model.rotary_emb.attention_scaling, 1.0)
        for name, value in model.state_dict().items():
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(value, weights[name]), name)

    def test_fp32_rope_rejects_different_type_scaling_or_original_buffer(self):
        c = config()
        c.head_dim = 128
        model = probe.Qwen3Model(c).to(dtype=torch.bfloat16)
        model.rotary_emb.rope_type = "linear"
        with self.assertRaisesRegex(ValueError, "default RoPE"):
            probe.restore_fp32_rope(model)
        model.rotary_emb.rope_type = "default"
        model.rotary_emb.attention_scaling = 2.0
        with self.assertRaisesRegex(ValueError, "scaling changed"):
            probe.restore_fp32_rope(model)
        model.rotary_emb.attention_scaling = 1.0
        model.rotary_emb.inv_freq[0] = 0
        with self.assertRaisesRegex(ValueError, "differs from R7"):
            probe.restore_fp32_rope(model)

    def test_summary_reports_original_mlx_alternative_and_eos_difference(self):
        logits = torch.zeros(151936, dtype=torch.bfloat16)
        logits[probe.EOS], logits[42474] = 24.875, 24.75
        result = probe.summary(logits)
        self.assertEqual(result["token_42474"], 24.75)
        self.assertEqual(result["eos_minus_token_42474"], 0.125)

    def test_checkpoint_mapping_and_chunked_cache(self):
        torch.manual_seed(17)
        original = probe.Qwen3Model(config()).to(dtype=torch.bfloat16).eval()
        original.config._attn_implementation = "sdpa"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            state = {probe.PREFIX + name: tensor for name, tensor in original.state_dict().items()}
            state["model.whisper_encoder.unused"] = torch.ones(1, dtype=torch.bfloat16)
            save_file(state, path)
            c = config()
            c._attn_implementation = "sdpa"
            loaded = probe.load_decoder(c, path)
            self.assertEqual(loaded.rotary_emb.inv_freq.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(loaded.rotary_emb.inv_freq, original.rotary_emb.inv_freq))
            for name, tensor in loaded.state_dict().items():
                self.assertTrue(torch.equal(tensor, original.state_dict()[name]))
            embeds = torch.randn(13, 16, dtype=torch.bfloat16)
            with torch.inference_mode():
                chunked, cache = probe.score(loaded, embeds, chunk_size=4)
                direct = loaded(inputs_embeds=embeds[None], use_cache=False).last_hidden_state[:, -1]
                direct_logits = torch.nn.functional.linear(direct, loaded.embed_tokens.weight)[0]
            self.assertEqual(cache.get_seq_length(), 13)
            self.assertEqual(chunked.dtype, torch.bfloat16)
            torch.testing.assert_close(chunked, direct_logits, atol=0.002, rtol=0.02)

    def test_pinned_moss_wrapper_equals_decoder_and_tied_head(self):
        source = os.environ.get("MOSS_OFFICIAL_SOURCE")
        if not source:
            self.skipTest("Set MOSS_OFFICIAL_SOURCE to the pinned upstream source directory")
        root = Path(source)
        sources = {
            "configuration_moss_transcribe_diarize": "b4d12b0f4609af69b61c2fe3aa5fbaf476af22278369e4540745bc47d1d37892",
            "modeling_moss_transcribe_diarize": "a01da90fe1f7cb88942b8c56f443e7b4ecd307ed4b4c356ac08d7503dd7422c1",
        }
        for name, expected in sources.items():
            self.assertEqual(probe.sha256(root / (name + ".py")), expected,
                             f"Pinned MOSS source changed: {name}")
        package = types.ModuleType("moss_probe_pinned")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        modules = {}
        for name in sources:
            qualified = package.__name__ + "." + name
            spec = importlib.util.spec_from_file_location(qualified, root / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            modules[name] = module
        cls = modules["configuration_moss_transcribe_diarize"].MossTranscribeDiarizeConfig
        c = cls(text_config=config().to_dict(), audio_config={"d_model":16, "encoder_layers":1,
                    "encoder_attention_heads":2, "encoder_ffn_dim":32, "max_source_positions":8,
                    "num_mel_bins":4, "decoder_layers":1, "decoder_attention_heads":2})
        c._attn_implementation = "sdpa"
        torch.manual_seed(17)
        full = modules["modeling_moss_transcribe_diarize"].MossTranscribeDiarizeForConditionalGeneration(c)
        full.to(dtype=torch.bfloat16).eval()
        self.assertIs(full.lm_head.weight, full.model.language_model.embed_tokens.weight)
        embeds = torch.randn(1, 9, 16, dtype=torch.bfloat16)
        with torch.inference_mode():
            official = full(inputs_embeds=embeds, logits_to_keep=1, use_cache=False).logits
            hidden = full.model.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state
            direct = torch.nn.functional.linear(hidden[:, -1:], full.model.language_model.embed_tokens.weight)
        self.assertTrue(torch.equal(official, direct))


if __name__ == "__main__":
    unittest.main()
