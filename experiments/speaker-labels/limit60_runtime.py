"""Audit the frozen fork through isolated oMLX without changing model arithmetic."""

from functools import wraps
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


PROMPT_SHA = "5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a"
DENOMINATOR_SHA = "831fa5a3736d5380f3eea7d8413c20240f9d56f7e3c04aa92533a16dcd5227f0"
PHASES = ("a-control", "a-hour", "b-control", "b-hour")


def audit(**values):
    print("LALIA_LIMIT60_AUDIT " + json.dumps(values), flush=True)


def check_arguments(argv, state):
    if len(argv) < 2 or argv[1] != "serve":
        raise ValueError("Only isolated serve is allowed")
    values = {}
    for flag in ("--host", "--port", "--base-path", "--model-dir"):
        if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv):
            raise ValueError(f"Require one {flag}")
        values[flag] = argv[argv.index(flag) + 1]
    base = Path(values["--base-path"]).resolve()
    allowed = {(state / phase / "server/MOSS-Transcribe-Diarize").resolve() for phase in PHASES}
    if ((values["--host"], values["--port"]) != ("127.0.0.1", "18001")
            or base not in allowed or Path(values["--model-dir"]).resolve() != base / "models"):
        raise ValueError("Require the frozen phase on loopback port 18001")


def checked_imports(state):
    modules = {}
    for name in ("mlx_audio", "mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize",
                 "mlx_audio.lm.generate", "omlx.engine.stt", "omlx.api.audio_models", "omlx.api.audio_routes"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to((state / "runtime").resolve()):
            raise ValueError(f"Import outside frozen runtime: {name}: {path}")
        modules[name] = module
    return modules


def install_audit(modules, prompt):
    import mlx.core as mx
    import numpy as np

    module = modules["mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize"]
    model_class = module.Model
    original_hook = model_class.post_load_hook.__func__
    original_generate = model_class.generate
    original_build = model_class._build_prompt
    step_module = modules["mlx_audio.lm.generate"]
    original_step = step_module.generate_step
    expected_prompt = prompt

    @classmethod
    @wraps(original_hook)
    def post_load_hook(cls, model, model_path):
        model = original_hook(cls, model, model_path)
        layers = model.layers
        if len(layers) != 28 or any(type(layer.self_attn.rope) is not module.MossRoPE for layer in layers):
            raise ValueError("Expected 28 constructor-created MossRoPE layers")
        hashes = {hashlib.sha256(np.array(layer.self_attn.rope._freqs).tobytes()).hexdigest()
                  for layer in layers}
        if hashes != {DENOMINATOR_SHA}:
            raise ValueError("Constructor frequency bytes differ")
        from mlx.utils import tree_flatten
        weights = [value for _, value in tree_flatten(model.parameters())]
        if not weights or any(value.dtype != mx.bfloat16 for value in weights):
            raise ValueError("Expected unquantized BF16 model weights")
        audit(loaded_layers=len(layers), denominator_sha256=DENOMINATOR_SHA, weights_dtype="bfloat16")
        return model

    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        if (kwargs.get("prompt") != prompt or kwargs.get("max_tokens") != 49152
                or kwargs.get("temperature", 0.0) != 0 or kwargs.get("repetition_penalty")
                or kwargs.get("stream") or kwargs.get("hotwords")
                or kwargs.get("prefill_step_size", 4096) != 4096):
            raise ValueError("Generation differs from the frozen request")
        audit(model_max_tokens=49152, prompt_sha256=PROMPT_SHA, eos_ids=sorted(self._eos_token_ids()))
        return original_generate(self, *args, **kwargs)

    @wraps(original_build)
    def build(self, audio_token_count, prompt=None):
        if prompt != expected_prompt:
            raise ValueError("Prompt builder received another instruction")
        result = original_build(self, audio_token_count, prompt)
        audit(builder_prompt_sha256=PROMPT_SHA, input_tokens=int(result.shape[-1]))
        return result

    @wraps(original_step)
    def step(*args, **kwargs):
        if (kwargs["max_tokens"] != 49152 or kwargs["prefill_step_size"] != 4096
                or kwargs.get("logits_processors")):
            raise ValueError("Generator budget, prefill or processors differ")
        audit(generator_max_tokens=49152, generator_input_tokens=int(kwargs["prompt"].shape[0]))
        count, last = 0, None
        source = original_step(*args, **kwargs)
        try:
            for token, probabilities in source:
                count, last = count + 1, int(token)
                yield token, probabilities
        finally:
            source.close()
            audit(yielded_tokens=count, last_token_id=last)

    model_class.post_load_hook = post_load_hook
    model_class.generate = generate
    model_class._build_prompt = build
    step_module.generate_step = step


def main():
    state = Path(os.environ["LALIA_LIMIT60_STATE"]).resolve(strict=True)
    if sys.argv[1:] == ["--check-imports"]:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        modules = checked_imports(state)
        prompt = (state / "prompt.txt").read_text()
        model = modules["mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize"].Model
        if modules["omlx.engine.stt"]._map_stt_prompt_kwargs(model, prompt) != {"prompt": prompt}:
            raise ValueError("Frozen oMLX cannot forward the official prompt to the fork")
        print(json.dumps({"status": "imports_only", "modules": {name: m.__file__ for name, m in modules.items()},
                          "installed_versions": {name: importlib.metadata.version(name) for name in
                                                 ("mlx", "mlx-metal", "mlx-audio", "torch", "transformers", "numpy")},
                          "model_loaded": False, "inference_run": False}))
        return
    check_arguments(sys.argv, state)
    prompt = (state / "prompt.txt").read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != PROMPT_SHA:
        raise ValueError("Frozen prompt changed")
    modules = checked_imports(state)
    audit(imports={name: module.__file__ for name, module in modules.items()})
    install_audit(modules, prompt)
    from omlx.cli import main as serve
    serve()


if __name__ == "__main__":
    main()
