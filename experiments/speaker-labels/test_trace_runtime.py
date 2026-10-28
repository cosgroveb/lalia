import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import trace_runtime as runtime
import prepare_trace


class TraceTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prompt = mx.array([7, 8], dtype=mx.int32)
        self.embeddings = mx.arange(8, dtype=mx.float32).reshape(2, 4).astype(mx.bfloat16)
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        self.expected = {"text_sha256": hashlib.sha256(b"1 2").hexdigest(),
                         "output_tokens": 2, "eos_id": 3}
        self.trace = runtime.Trace(self.root / "trace", self.expected, (2, 4))

    def kwargs(self):
        return {"prompt": self.prompt, "input_embeddings": self.embeddings,
                "model": SimpleNamespace(_eos_token_ids=lambda: {3}),
                "max_tokens": 49152, "prefill_step_size": 4096, "sampler": object()}

    def test_exact_yielded_vector_survives_lookahead_and_closes(self):
        kwargs = self.kwargs()
        vectors = [mx.array([-5, 0, -3, -4]), mx.array([-6, -4, 0, -3]), mx.array([-7, -4, -2, 0])]
        closed = []
        lookahead = []

        def original(*args, **received):
            self.assertEqual(args, ("sentinel",))
            self.assertIs(received["sampler"], kwargs["sampler"])
            self.assertIs(received["prompt"], self.prompt)
            try:
                for token, vector in zip((1, 2, 3), vectors):
                    lookahead.append(mx.array([0, -9, -9, -9]))
                    yield token, vector
            finally:
                closed.append(True)

        observed = runtime.observe_step(original, self.trace, ("sentinel",), kwargs)
        for token, vector in observed:
            self.assertIs(vector, vectors[token - 1])
            if token == 3:
                break
        observed.close()
        self.assertEqual(closed, [True])
        self.assertEqual(self.trace.tokens, [1, 2, 3])
        saved = mx.load(str(self.trace.directory / "eos-logprobs.safetensors"))["logprobs"]
        self.assertTrue(bool(mx.array_equal(saved, vectors[-1]).item()))
        self.assertFalse(bool(mx.array_equal(saved, lookahead[-1]).item()))
        result = SimpleNamespace(text="1 2", generation_tokens=2, prompt_tokens=2)
        lookup = lambda ids: mx.broadcast_to(ids[:, None], (len(ids), 4)).astype(mx.bfloat16)
        summary = self.trace.finish(result, self.tokenizer, lookup)
        self.assertTrue(summary["reproduces_previous"])
        combined = mx.load(str(self.trace.directory / "combined-embeddings.safetensors"))["inputs_embeds"]
        self.assertEqual(combined.shape, (4, 4))
        self.assertTrue(bool(mx.array_equal(combined[:2], self.embeddings).item()))
        self.assertEqual(summary["combined_embeddings"]["float32_content_sha256"],
                         hashlib.sha256(np.array(combined.astype(mx.float32)).tobytes()).hexdigest())
        self.assertEqual(summary["inputs"]["prompt_ids_raw_sha256"],
                         hashlib.sha256(np.array(self.prompt).tobytes()).hexdigest())

    def test_cancellation_preserves_partial_ids_and_closes(self):
        closed = []

        def original(**kwargs):
            try:
                yield 1, mx.array([-1, 0, -2, -3])
                yield 3, mx.array([-4, -2, -1, 0])
            finally:
                closed.append(True)

        observed = runtime.observe_step(original, self.trace, (), self.kwargs())
        next(observed)
        observed.close()
        self.assertEqual(closed, [True])
        status = json.loads((self.trace.directory / "status.json").read_text())
        self.assertEqual(status["status"], "incomplete")
        self.assertEqual(json.loads((self.trace.directory / "token-ids.json").read_text()), [1])

    def test_input_rejection_and_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            runtime.Trace(self.trace.directory, self.expected)
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            self.trace.capture_inputs(self.prompt, self.embeddings.astype(mx.float32))
        self.trace.capture_inputs(self.prompt, self.embeddings)
        with self.assertRaisesRegex(ValueError, "already captured"):
            self.trace.capture_inputs(self.prompt, self.embeddings)
        saved = mx.load(str(self.trace.directory / "inputs.safetensors"))
        self.assertTrue(bool(mx.array_equal(saved["prompt_ids"], self.prompt).item()))
        self.assertTrue(bool(mx.array_equal(saved["input_embeddings"], self.embeddings).item()))

    def test_missing_eos_and_returned_text_mismatch_are_not_complete(self):
        self.trace.capture_inputs(self.prompt, self.embeddings)
        with self.assertRaisesRegex(ValueError, "native EOS"):
            self.trace.finish(SimpleNamespace(), self.tokenizer, None)
        self.trace.record(3, mx.array([-3, -2, -1, 0]), {3})
        result = SimpleNamespace(text="wrong", generation_tokens=0, prompt_tokens=2)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.trace.finish(result, self.tokenizer, None)
        self.assertEqual(json.loads((self.trace.directory / "status.json").read_text())["status"], "started")

    def test_trace_does_not_force_previous_reproduction(self):
        self.trace.capture_inputs(self.prompt, self.embeddings)
        self.trace.record(1, mx.array([-3, 0, -1, -2]), {3})
        self.trace.record(3, mx.array([-3, -2, -1, 0]), {3})
        result = SimpleNamespace(text="1", generation_tokens=1, prompt_tokens=2)
        summary = self.trace.finish(result, self.tokenizer, lambda ids: mx.zeros((len(ids), 4), mx.bfloat16))
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["reproduces_previous"])

    def test_generator_failure_propagates_and_marks_incomplete(self):
        def original(**kwargs):
            yield 1, mx.array([-3, 0, -1, -2])
            raise RuntimeError("upstream failure")

        with self.assertRaisesRegex(RuntimeError, "upstream failure"):
            list(runtime.observe_step(original, self.trace, (), self.kwargs()))
        self.assertEqual(json.loads((self.trace.directory / "status.json").read_text())["status"], "incomplete")

    def test_frozen_limits_and_single_generator_call(self):
        kwargs = self.kwargs()
        kwargs["prefill_step_size"] = 2048
        with self.assertRaisesRegex(ValueError, "limits differ"):
            list(runtime.observe_step(None, self.trace, (), kwargs))
        with self.assertRaisesRegex(ValueError, "one generator"):
            list(runtime.observe_step(None, self.trace, (), self.kwargs()))


class WiringTests(unittest.TestCase):
    def test_model_wrapper_completes_only_after_returned_text_validation(self):
        mx.set_default_device(mx.cpu)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trace"
            tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids)))
            closed = []

            def original_step(**kwargs):
                try:
                    for token in (1, 2, 3):
                        yield token, mx.array([-4, -3, -2, 0])
                finally:
                    closed.append(True)

            step_module = SimpleNamespace(generate_step=original_step)

            class Model:
                _tokenizer = tokenizer
                model = SimpleNamespace(language_model=SimpleNamespace(
                    embed_tokens=lambda ids: mx.zeros((len(ids), 4), mx.bfloat16)))

                def _eos_token_ids(self):
                    return {3}

                def generate(self, audio, **kwargs):
                    captured = []
                    for token, _ in step_module.generate_step(
                            prompt=mx.array([7, 8]), input_embeddings=mx.zeros((2, 4), mx.bfloat16),
                            model=self, max_tokens=kwargs["max_tokens"], prefill_step_size=4096):
                        if token == 3:
                            break
                        captured.append(token)
                    return SimpleNamespace(text=self._tokenizer.decode(captured), generation_tokens=len(captured),
                                           prompt_tokens=2)

            expected = {"text_sha256": hashlib.sha256(b"1 2").hexdigest(), "output_tokens": 2, "eos_id": 3}
            original_trace = runtime.Trace
            with patch.object(runtime, "Trace", side_effect=lambda path, expected: original_trace(path, expected, (2, 4))), \
                    patch.object(runtime, "PROMPT_SHA", hashlib.sha256(b"fixture").hexdigest()), \
                    contextlib.redirect_stdout(io.StringIO()):
                runtime.install_observation(Model, step_module, directory, expected)
                result = Model().generate("audio", max_tokens=49152, prompt="fixture")
                self.assertEqual(result.text, "1 2")
                self.assertEqual(closed, [True])
                self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "complete")
                with self.assertRaises(FileExistsError):
                    Model().generate("audio", max_tokens=49152, prompt="fixture")

    def test_only_candidate_phases_install_rope(self):
        calls = []
        factory = object()
        with patch.dict(sys.modules, {"eos_rope": SimpleNamespace(NativeBF16RoPE=factory)}), \
                patch.object(runtime, "install_hook", side_effect=lambda cls, factory: calls.append((cls, factory))):
            runtime.apply_rope(str, "original-1")
            self.assertEqual(calls, [])
            runtime.apply_rope(str, "candidate-1")
            runtime.apply_rope(str, "candidate-2")
        self.assertEqual(calls, [(str, factory), (str, factory)])

    def test_only_fresh_states_and_loopback_port_are_allowed(self):
        state = runtime.TRACE_STATES / "test-run"
        for phase in runtime.PHASES:
            base = state / phase / "server/MOSS-Transcribe-Diarize"
            argv = ["wrapper", "serve", "--host", "127.0.0.1", "--port", "18001",
                    "--base-path", str(base), "--model-dir", str(base / "models")]
            self.assertEqual(runtime.checked_phase(argv, state), phase)
            argv[argv.index("--port") + 1] = "8000"
            with self.assertRaises(ValueError):
                runtime.checked_phase(argv, state)
        with self.assertRaisesRegex(ValueError, "fresh state"):
            runtime.checked_phase(argv, runtime.BASE.parent / "exact-trace-20261027")

    def test_post_load_hook_checks_architecture_and_frequency_provenance(self):
        calls = []

        class Model:
            @classmethod
            def post_load_hook(cls, model, path):
                calls.append((cls, model, path))
                return model

        config = SimpleNamespace(head_dim=128, rope_theta=1000000,
                                 max_position_embeddings=131072, num_hidden_layers=28)
        layers = [SimpleNamespace(self_attn=SimpleNamespace(head_dim=128, rope=None)) for _ in range(28)]
        model = SimpleNamespace(config=SimpleNamespace(text_config=config), layers=layers)
        rope = SimpleNamespace(count=131072, frequencies={"denominator_sha256": runtime.DENOMINATOR_SHA})
        factory = unittest.mock.Mock(return_value=rope)
        runtime.install_hook(Model, factory)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(Model.post_load_hook(model, "/model"), model)
        self.assertEqual(calls, [(Model, model, "/model")])
        factory.assert_called_once_with(131072)
        self.assertTrue(all(layer.self_attn.rope is rope for layer in layers))
        rope.frequencies["denominator_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            Model.post_load_hook(model, "/model")
        config.head_dim = 64
        with self.assertRaisesRegex(ValueError, "rotary configuration"):
            Model.post_load_hook(model, "/model")

    def test_preparation_refuses_historical_or_existing_state(self):
        with self.assertRaisesRegex(ValueError, "fresh state"):
            prepare_trace.prepare(runtime.BASE.parent / "exact-trace-20261027")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "existing"
            state.mkdir()
            with patch.object(prepare_trace, "TRACE_STATES", root):
                with self.assertRaisesRegex(ValueError, "State already exists"):
                    prepare_trace.prepare(state)

    def test_preparation_pins_integrated_sources_and_sets_wrapper_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = root / "official-api-path"
            previous = base / "native-bf16-api-20261027"
            previous.mkdir(parents=True)
            command = ["python", "run.py", "--state", "old", "--omlx", "old", "--baseline", "old"]
            (previous / "protocol.json").write_text(json.dumps({"hashes": {}, "commands": {"stress90": command},
                "model_revision": "fixture", "versions": {}, "limits": {}}))
            (previous / "protocol.sha256").write_text("fixture")
            for historical in (base, previous):
                result = historical / "stress90/results/MOSS-Transcribe-Diarize/ami-ES2004a-stress-90m"
                result.mkdir(parents=True)
                (result / "response.json").write_text(json.dumps({"text": "public fixture", "usage": {"output_tokens": 2}}))
                (result / "result.json").write_text("{}")
                baseline = historical / "baseline/results/MOSS-Transcribe-Diarize/ami-ES2004a-full"
                baseline.mkdir(parents=True)
                (baseline / "result.json").write_text("{}")
                server = historical / "stress90/server/MOSS-Transcribe-Diarize"
                server.mkdir(parents=True)
                (server / "server.log").write_text('LALIA_API_AUDIT {"last_token_id": 151645, "yielded_tokens": 3}\n')
            model = base / "baseline/models/MOSS-Transcribe-Diarize"
            model.mkdir(parents=True)
            states = root / "exact-traces"
            state = states / "fresh"
            original_sha = runtime.sha256
            def digest(path):
                if Path(path) == previous / "protocol.json":
                    return "a48c761b408eb2ffc484df99a343b139fb2ff86e609f4f730f06531424865dd9"
                return original_sha(path)
            with patch.object(prepare_trace, "BASE", base), patch.object(prepare_trace, "TRACE_STATES", states), \
                    patch.object(prepare_trace, "sha256", side_effect=digest), \
                    patch.object(prepare_trace.subprocess, "run", return_value=SimpleNamespace(stdout="AC")):
                result = prepare_trace.prepare(state)
            self.assertFalse(result["inference_run"])
            self.assertEqual(result["protocol_sha256"], original_sha(state / "protocol.json"))
            protocol = json.loads((state / "protocol.json").read_text())
            source = Path(runtime.__file__).parent
            self.assertEqual(protocol["hashes"][str(source / "eos_rope.py")], original_sha(source / "eos_rope.py"))
            for phase in runtime.PHASES:
                argv = protocol["commands"][phase]
                self.assertEqual(argv[:2], ["env", f"LALIA_EXACT_TRACE_STATE={state}"])
                self.assertEqual(argv[argv.index("--omlx") + 1], str(source / "omlx-exact-trace"))
                self.assertEqual(argv[argv.index("--state") + 1], str(state / phase))
                self.assertEqual((state / phase / "models/MOSS-Transcribe-Diarize").resolve(), model)


if __name__ == "__main__":
    unittest.main()
