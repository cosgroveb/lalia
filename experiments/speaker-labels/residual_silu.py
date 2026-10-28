"""Model-local SiLU arithmetic comparison. Import only inside a reviewed worker."""

from functools import partial
import hashlib
import importlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

TEXT_MODULE = "mlx_audio.stt.models.qwen3_asr.qwen3_asr"
TEXT_SOURCE_SHA = "661e6e930842dcd0cd509a75964f4b0fcf4d935a52011062d2978f050bc6b38d"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
IDENTITY_KEYS = (*PROJECTIONS, *(name + ".weight" for name in PROJECTIONS), "historical_call", "helper")
DTYPE_KEYS = ("gate", "up", "historical_activation", "candidate_activation", "explicit_sigmoid",
              "explicit_silu", "historical_product", "candidate_product", "explicit_product")
NATIVE_SHA = "5792f70d8d82c402ddc1cb724577cffb196d0d0bd5fad159b94ccde313b24416"
CANDIDATE_SHA = "c85ea2966b5c1b3bededbb755b8c99e2f84ae7d027e0ae88c4067f46593e9bd8"


@partial(mx.compile, shapeless=True)
def full_silu_bf16(gate):
    gate32 = gate.astype(mx.float32)
    return (gate32 / (1 + mx.exp(-gate32))).astype(mx.bfloat16)


def require_bf16(**values):
    if any(value.dtype != mx.bfloat16 for value in values.values()):
        raise ValueError("Require BF16: " + ", ".join(values))


class FullSiLUTextMLP(nn.Module):
    activation = staticmethod(full_silu_bf16)

    def __init__(self, original):
        super().__init__()
        self.hidden_size = original.hidden_size
        self.intermediate_size = original.intermediate_size
        for name in PROJECTIONS:
            setattr(self, name, getattr(original, name))
        self._training = original.training
        self._no_grad = set(original._no_grad)

    def __call__(self, x):
        gate = self.gate_proj(x)
        require_bf16(gate=gate)
        activation = self.activation(gate)
        up = self.up_proj(x)
        product = activation * up
        require_bf16(up=up, activation=activation, product=product)
        result = self.down_proj(product)
        require_bf16(down=result)
        return result


def historical_text(runtime):
    module = importlib.import_module(TEXT_MODULE)
    filename = Path(module.__file__).resolve()
    expected = runtime.resolve() / "mlx_audio/stt/models/qwen3_asr/qwen3_asr.py"
    if filename != expected or hashlib.sha256(filename.read_bytes()).hexdigest() != TEXT_SOURCE_SHA:
        raise ValueError("TextMLP imported outside pinned historical source")
    if module.TextMLP.__module__ != TEXT_MODULE or module.TextModel.__module__ != TEXT_MODULE:
        raise ValueError("Unexpected TextMLP/TextModel class origin")
    if module.TextMLP.__call__.__globals__["nn"].silu is not nn.silu:
        raise ValueError("Historical TextMLP activation differs")
    return module


def select_activation(model, arm, runtime):
    """Keep baseline MLPs or replace only the candidate's historical TextMLPs."""
    if arm not in ("baseline", "candidate"):
        raise ValueError("Unknown SiLU arm")
    module = historical_text(runtime)
    language = model.model.language_model
    if type(language) is not module.TextModel or len(language.layers) != 28:
        raise ValueError("Require historical 28-layer TextModel")
    originals = [layer.mlp for layer in language.layers]
    for original in originals:
        if type(original) is not module.TextMLP:
            raise ValueError("Require original historical TextMLP in all 28 layers")
        if (original.hidden_size, original.intermediate_size) != (1024, 3072):
            raise ValueError("Unexpected TextMLP dimensions")
        require_bf16(**{name: getattr(original, name).weight for name in PROJECTIONS})
    parameters = dict(tree_flatten(model.parameters()))
    if arm == "candidate":
        replacements = [FullSiLUTextMLP(original) for original in originals]
        for layer, replacement in zip(language.layers, replacements):
            layer.mlp = replacement
    for original, layer in zip(originals, language.layers):
        for name in PROJECTIONS:
            if getattr(layer.mlp, name) is not getattr(original, name):
                raise ValueError("Projection identity changed")
    after = dict(tree_flatten(model.parameters()))
    if parameters.keys() != after.keys() or any(value is not after[name] for name, value in parameters.items()):
        raise ValueError("Parameter paths or identities changed")
    if FullSiLUTextMLP.activation is not full_silu_bf16:
        raise ValueError("Candidate activation helper changed")
    return {"arm": arm, "source": str(Path(module.__file__).resolve()),
            "source_sha256": TEXT_SOURCE_SHA, "original_class": TEXT_MODULE + ".TextMLP",
            "replacement_count": 28 if arm == "candidate" else 0,
            "projection_identities_retained": True, "parameter_identities_retained": True,
            "parameter_paths": sorted(parameters), "helper_identity_verified": True,
            "weight_dtype": "mlx.core.bfloat16",
            "forward_dtype_checks": "gate, up, activation, product, down" if arm == "candidate" else
                                    "Unobserved. Original instances retained, exact replay vectors required."}


def fixed_cpu_reference():
    """The original check.py fixture and Torch CPU calculation, without MLX work."""
    import numpy as np
    import torch

    gate = torch.linspace(-16, 16, 8193).bfloat16().unique().float().numpy()
    up = np.resize(np.array([-3., -.5, .25, 1., 2.], np.float32), gate.size)
    reference = (torch.nn.functional.silu(torch.from_numpy(gate).bfloat16())
                 * torch.from_numpy(up).bfloat16()).float().numpy()
    return gate, up, reference


def compare_cpu(value, reference):
    import numpy as np

    return {"different_from_cpu_torch": int(np.count_nonzero(value != reference)),
            "max_abs_error": float(np.max(np.abs(value - reference))),
            "fp32_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def require_preflight(report):
    if report["elements"] != 1537 or report["domain"] != [-16, 16]:
        raise ValueError("SiLU preflight fixture differs")
    if report["mlx"] != "0.32.4" or report["torch"] != "2.14.1" or report["device"] != "Device(gpu, 0)":
        raise ValueError("SiLU preflight runtime differs")
    if report["identity_checks"] != dict.fromkeys(IDENTITY_KEYS, True):
        raise ValueError("SiLU preflight identity check failed")
    if report["dtypes"] != dict.fromkeys(DTYPE_KEYS, "mlx.core.bfloat16"):
        raise ValueError("SiLU preflight dtype check failed")
    if report["historical_vs_explicit_differences"] != 0:
        raise ValueError("Historical TextMLP differs from explicit BF16 boundaries")
    for name, digest, count, error in (("historical", NATIVE_SHA, 458, 0.125),
                                        ("explicit", NATIVE_SHA, 458, 0.125),
                                        ("candidate", CANDIDATE_SHA, 0, 0.0)):
        expected = {"fp32_sha256": digest, "different_from_cpu_torch": count, "max_abs_error": error}
        if report["comparisons"][name] != expected:
            raise ValueError(f"SiLU preflight {name} did not reproduce pinned output")
    if report["cpu_reference_fp32_sha256"] != CANDIDATE_SHA:
        raise ValueError("SiLU preflight CPU reference differs")


def operator_preflight(directory, runtime):
    """One later supervised GPU worker. No CLI, model load, or retry here."""
    import numpy as np
    import torch

    if mx.default_device() != mx.gpu:
        raise ValueError("Exact-path preflight requires the reviewed Metal worker")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    module = historical_text(runtime)
    gate_np, up_np, reference = fixed_cpu_reference()
    gate = mx.array(gate_np).astype(mx.bfloat16)
    up = mx.array(up_np).astype(mx.bfloat16)

    class FixedProjection(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = value

        def __call__(self, _):
            return self.weight

    class IdentityProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array([1], dtype=mx.bfloat16)

        def __call__(self, value):
            require_bf16(down_input=value)
            return value

    historical = module.TextMLP.__new__(module.TextMLP)
    nn.Module.__init__(historical)
    historical.hidden_size, historical.intermediate_size = 1024, 3072
    historical.gate_proj, historical.up_proj = FixedProjection(gate), FixedProjection(up)
    historical.down_proj = IdentityProjection()
    candidate = FullSiLUTextMLP(historical)
    identity_checks = {name: getattr(candidate, name) is getattr(historical, name) for name in PROJECTIONS}
    identity_checks.update({name + ".weight": getattr(candidate, name).weight is getattr(historical, name).weight
                            for name in PROJECTIONS})
    identity_checks.update(historical_call=type(historical).__call__ is module.TextMLP.__call__,
                           helper=candidate.activation is full_silu_bf16)
    native = historical(gate)
    changed = candidate(gate)
    # Observe each activation's dtype separately without replacing either callable.
    native_activation, candidate_activation = nn.silu(gate), full_silu_bf16(gate)
    sigmoid = mx.sigmoid(gate)
    mx.eval(sigmoid)
    explicit_silu = gate * sigmoid
    mx.eval(explicit_silu)
    explicit = explicit_silu * up
    values = {"gate": gate, "up": up, "historical_activation": native_activation,
              "candidate_activation": candidate_activation, "explicit_sigmoid": sigmoid,
              "explicit_silu": explicit_silu, "historical_product": native,
              "candidate_product": changed, "explicit_product": explicit}
    require_bf16(**values)
    mx.eval(*values.values())
    arrays = {name: np.asarray(value.astype(mx.float32)) for name, value in
              (("historical", native), ("candidate", changed), ("explicit", explicit))}
    np.savez(directory / "vectors.npz", gate=gate_np, up=up_np, cpu_reference=reference, **arrays)
    report = {"mlx": mx.__version__, "torch": torch.__version__, "device": str(mx.default_device()),
              "elements": int(gate_np.size), "domain": [-16, 16],
              "reference": "local Torch CPU F.silu times up, not CUDA evidence",
              "source": str(Path(module.__file__).resolve()), "source_sha256": TEXT_SOURCE_SHA,
              "identity_checks": identity_checks, "dtypes": {name: str(value.dtype) for name, value in values.items()},
              "comparisons": {name: compare_cpu(value, reference) for name, value in arrays.items()},
              "historical_vs_explicit_differences": int(np.count_nonzero(arrays["historical"] != arrays["explicit"])),
              "cpu_reference_fp32_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
              "vectors_sha256": hashlib.sha256((directory / "vectors.npz").read_bytes()).hexdigest(),
              "status": "failed"}
    try:
        require_preflight(report)
        report["status"] = "complete"
    finally:
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
