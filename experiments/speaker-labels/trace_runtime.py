"""Observe exact MOSS generator inputs and the vector paired with native EOS."""

from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


BASE = Path.home() / ".local/state/lalia-speaker-qualification/official-api-path"
TRACE_STATES = BASE.parent / "exact-traces"
DENOMINATOR_SHA = "831fa5a3736d5380f3eea7d8413c20240f9d56f7e3c04aa92533a16dcd5227f0"
PHASES = {"candidate-1": True, "candidate-2": True, "original-1": False}
PROMPT_SHA = "5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_arrays(path, arrays):
    import mlx.core as mx

    temporary = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(temporary), arrays)
    temporary.replace(path)
    return {"sha256": sha256(path), "arrays": {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()}}


def float32_digest(value):
    import mlx.core as mx
    import numpy as np

    digest = hashlib.sha256()
    for start in range(0, value.shape[0], 2048):
        with mx.stream(mx.cpu):
            chunk = value[start:start + 2048].astype(mx.float32)
            mx.eval(chunk)
            digest.update(np.array(chunk).tobytes())
    return digest.hexdigest()


def checked_phase(argv, state):
    if state.resolve().parent != TRACE_STATES.resolve():
        raise ValueError("Require a fresh state under exact-traces")
    if len(argv) < 2 or argv[1] != "serve":
        raise ValueError("Only isolated serve is allowed")
    values = {}
    for flag in ("--host", "--port", "--base-path", "--model-dir"):
        if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv):
            raise ValueError(f"Require one {flag} value")
        values[flag] = argv[argv.index(flag) + 1]
    if (values["--host"], values["--port"]) != ("127.0.0.1", "18001"):
        raise ValueError("Require loopback port 18001")
    base = Path(values["--base-path"]).resolve()
    for phase in PHASES:
        expected = (state / phase / "server/MOSS-Transcribe-Diarize").resolve()
        if base == expected and Path(values["--model-dir"]).resolve() == expected / "models":
            return phase
    raise ValueError("Require a fresh exact-trace state")


class Trace:
    def __init__(self, directory, expected, dimensions=(71687, 1024)):
        self.directory = directory
        directory.mkdir(mode=0o700)
        self.expected = expected
        self.dimensions = dimensions
        self.tokens = []
        self.eos = None
        self.inputs = None
        self.step_started = False
        save(directory / "status.json", {"status": "started"})

    def capture_inputs(self, prompt, embeddings):
        import mlx.core as mx
        import numpy as np

        if self.inputs is not None:
            raise ValueError("Generator inputs already captured")
        if (tuple(embeddings.shape) != self.dimensions or embeddings.dtype != mx.bfloat16
                or tuple(prompt.shape) != (self.dimensions[0],)
                or prompt.dtype not in (mx.int32, mx.int64)):
            raise ValueError("Unexpected generator input shape or dtype")
        self.inputs = save_arrays(self.directory / "inputs.safetensors", {
            "prompt_ids": prompt, "input_embeddings": embeddings})
        self.inputs["prompt_ids_raw_sha256"] = hashlib.sha256(np.array(prompt).tobytes()).hexdigest()
        save(self.directory / "inputs.json", self.inputs)

    def record(self, token, logprobs, eos_ids):
        import mlx.core as mx

        if self.eos is not None:
            raise ValueError("Generator yielded after EOS")
        self.tokens.append(int(token))
        if len(self.tokens) % 1024 == 0:
            save(self.directory / "progress.json", {"yielded_tokens": len(self.tokens)})
        if int(token) in eos_ids:
            if logprobs.ndim != 1 or not bool(mx.all(mx.isfinite(logprobs)).item()):
                raise ValueError("Invalid EOS log-probability vector")
            arrays = save_arrays(self.directory / "eos-logprobs.safetensors", {"logprobs": logprobs})
            values = logprobs.astype(mx.float32)
            allowed = mx.ones(values.shape, dtype=mx.bool_)
            for eos_id in eos_ids:
                allowed[eos_id] = False
            continuation = int(mx.argmax(mx.where(allowed, values, -mx.inf)).item())
            self.eos = {
                "token_id": int(token), "output_tokens": len(self.tokens) - 1,
                "yielded_tokens": len(self.tokens), "argmax_id": int(mx.argmax(logprobs).item()),
                "eos_logprob": float(values[int(token)].item()),
                "best_continuation_id": continuation,
                "best_continuation_logprob": float(values[continuation].item()),
                "eos_margin": float((values[int(token)] - values[continuation]).item()),
                "artifact": arrays,
            }
            save(self.directory / "eos.json", self.eos)
            save(self.directory / "token-ids.json", self.tokens)

    def finish(self, result, tokenizer, embedding_lookup):
        import mlx.core as mx

        if self.inputs is None or self.eos is None:
            raise ValueError("Generation returned without complete inputs and native EOS")
        text = tokenizer.decode(self.tokens[:-1], skip_special_tokens=True).strip()
        if (text != result.text or result.generation_tokens != len(self.tokens) - 1
                or result.prompt_tokens != self.dimensions[0]):
            raise ValueError("Captured tokens disagree with returned transcription")
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        observed = {"text_sha256": text_sha, "output_tokens": len(self.tokens) - 1,
                    "eos_id": self.eos["token_id"]}
        prompt = mx.load(str(self.directory / "inputs.safetensors"))["input_embeddings"]
        generated = embedding_lookup(mx.array(self.tokens[:-1], dtype=mx.int32))
        if generated.dtype != mx.bfloat16 or tuple(generated.shape) != (len(self.tokens) - 1, self.dimensions[1]):
            raise ValueError("Generated-token embeddings differ from prompt dtype/dimensions")
        embeddings = mx.concatenate([prompt, generated], axis=0)
        combined = save_arrays(self.directory / "combined-embeddings.safetensors", {"inputs_embeds": embeddings})
        combined["float32_content_sha256"] = float32_digest(embeddings)
        summary = {"status": "complete", "observed": observed, "expected": self.expected,
                   "reproduces_previous": observed == self.expected,
                   "token_ids_sha256": sha256(self.directory / "token-ids.json"),
                   "inputs": self.inputs, "combined_embeddings": combined, "eos": self.eos}
        save(self.directory / "status.json", summary)
        return summary

    def fail(self, error):
        save(self.directory / "token-ids.json", self.tokens)
        save(self.directory / "status.json", {"status": "incomplete", "error": type(error).__name__,
                                               "yielded_tokens": len(self.tokens), "eos_seen": self.eos is not None})


def observe_step(original, trace, args, kwargs):
    if trace.step_started:
        raise ValueError("Only one generator call is allowed")
    trace.step_started = True
    if (kwargs.get("max_tokens"), kwargs.get("prefill_step_size")) != (49152, 4096):
        raise ValueError("Generation limits differ from the frozen API call")
    trace.capture_inputs(kwargs["prompt"], kwargs["input_embeddings"])
    eos_ids = kwargs["model"]._eos_token_ids()
    generator = original(*args, **kwargs)
    try:
        for token, logprobs in generator:
            trace.record(token, logprobs, eos_ids)
            yield token, logprobs
    finally:
        generator.close()
        if trace.eos is None:
            trace.fail(RuntimeError("Generator closed before native EOS"))


def install_observation(model_class, step_module, directory, expected):
    original_generate = model_class.generate
    original_step = step_module.generate_step
    active = None

    @wraps(original_step)
    def step(*args, **kwargs):
        if active is None:
            raise ValueError("Generator called outside the traced request")
        yield from observe_step(original_step, active, args, kwargs)

    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        nonlocal active
        prompt = kwargs.get("prompt")
        if (not isinstance(prompt, str) or hashlib.sha256(prompt.encode()).hexdigest() != PROMPT_SHA
                or kwargs.get("max_tokens") != 49152 or kwargs.get("stream") or kwargs.get("hotwords")):
            raise ValueError("Request differs from the frozen API call")
        active = Trace(directory, expected)
        try:
            result = original_generate(self, *args, **kwargs)
            summary = active.finish(result, self._tokenizer, self.model.language_model.embed_tokens)
            print("LALIA_EXACT_TRACE " + json.dumps({
                "status": summary["status"], "observed": summary["observed"],
                "reproduces_previous": summary["reproduces_previous"],
                "eos_margin": summary["eos"]["eos_margin"]}), flush=True)
            return result
        except BaseException as error:
            active.fail(error)
            raise
        finally:
            active = None

    model_class.generate = generate
    step_module.generate_step = step


def install_hook(model_class, rope_factory):
    original = model_class.post_load_hook.__func__

    @classmethod
    @wraps(original)
    def post_load_hook(cls, model, model_path):
        model = original(cls, model, model_path)
        config = model.config.text_config
        if (config.head_dim, config.rope_theta, config.max_position_embeddings,
                config.num_hidden_layers) != (128, 1000000, 131072, 28):
            raise ValueError("Unexpected MOSS rotary configuration")
        layers = model.layers
        if len(layers) != 28 or any(layer.self_attn.head_dim != 128 for layer in layers):
            raise ValueError("Unexpected MOSS attention layers")
        rope = rope_factory(131072)
        if rope.frequencies["denominator_sha256"] != DENOMINATOR_SHA:
            raise ValueError("Native RoPE denominators differ from the measured candidate")
        for layer in layers:
            layer.self_attn.rope = rope
        print("LALIA_API_AUDIT " + json.dumps({
            "native_bf16_rope_layers": len(layers),
            "native_bf16_rope_positions": rope.count,
            "native_bf16_rope_frequencies": rope.frequencies,
        }), flush=True)
        return model

    model_class.post_load_hook = post_load_hook


def apply_rope(model_class, phase):
    if PHASES[phase]:
        from eos_rope import NativeBF16RoPE
        install_hook(model_class, NativeBF16RoPE)


def main():
    os.umask(0o077)
    state = Path(os.environ["LALIA_EXACT_TRACE_STATE"]).resolve()
    phase = checked_phase(sys.argv, state)
    protocol_path = state / "protocol.json"
    if sha256(protocol_path) != (state / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["hashes"].items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    from mlx_audio.stt.models.moss_transcribe_diarize import Model

    module = Path(importlib.import_module(Model.__module__).__file__).resolve()
    if not module.is_relative_to((BASE / "runtime").resolve()):
        raise ValueError("MOSS imported outside the isolated runtime")
    apply_rope(Model, phase)
    install_observation(Model, importlib.import_module("mlx_audio.lm.generate"),
                        state / phase / "trace", protocol["expected"][phase])
    from omlx.cli import main as serve
    serve()


if __name__ == "__main__":
    main()
