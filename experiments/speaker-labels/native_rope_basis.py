"""Tiny native RoPE basis check. No model weights or transcript generation."""

import argparse
import importlib.metadata
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from eos_rope import NativeBF16RoPE, ReferenceRoPE, TOTAL
from window_matching import save, sha256


def check(result):
    # A gross-error check, not a claim of arithmetic parity with Torch eager.
    for row in result["cases"]:
        if not row["finite"] or row["max_abs_error"] > 0.02:
            raise ValueError(f"Native basis mismatch: {row}")
        if row["offset"] == 0 and not row["position_zero_exact"]:
            raise ValueError("Position zero is not identity")
        if row["offset"] and row["wrong_denominator_max_abs_error"] < 0.5:
            raise ValueError("Basis does not detect inverted denominators")


def measure():
    torch.set_num_threads(2)
    original_device = mx.default_device()
    rows, outputs = [], {}
    try:
        mx.set_default_device(mx.gpu)
        native = NativeBF16RoPE(TOTAL + 3)
        reference = ReferenceRoPE(TOTAL + 3, bf16_frequencies=True)
        for dtype in (mx.float32, mx.bfloat16):
            for length in (1, 4):
                values = np.zeros((1, length, 16, 128), dtype=np.float32)
                values[:, :, :8, :64] = 1
                values[:, :, 8:, 64:] = 1
                # Match Q's projection layout, then transpose heads and tokens.
                value = mx.array(values, dtype=dtype).transpose(0, 2, 1, 3)
                for offset in (0, 88813):
                    actual = native(value, offset)
                    expected = reference(value, offset)
                    wrong = mx.fast.rope(value, 128, traditional=False, base=None, scale=1.0,
                                         offset=offset, freqs=mx.reciprocal(native.freqs))
                    mx.eval(actual, expected, wrong)
                    left = np.array(actual.astype(mx.float32))
                    right = np.array(expected.astype(mx.float32))
                    difference = left.astype(np.float64) - right
                    name = f"{str(dtype).split('.')[-1]}_{length}_{offset}"
                    outputs[name + "_native"] = actual
                    outputs[name + "_reference"] = expected
                    rows.append({"dtype": str(dtype), "shape": list(value.shape), "offset": offset,
                                 "expected_kernel": "rope_" + ("single_" if length == 1 else "") + "freqs_" + str(dtype).split(".")[-1],
                                 "finite": bool(np.isfinite(left).all()),
                                 "max_abs_error": float(np.max(np.abs(difference))),
                                 "rms_error": float(np.sqrt(np.mean(difference ** 2))),
                                 "different_elements": int(np.count_nonzero(left != right)),
                                 "position_zero_exact": bool(np.array_equal(left[:, :, 0], np.array(value.astype(mx.float32))[:, :, 0])) if offset == 0 else None,
                                 "wrong_denominator_max_abs_error": float(np.max(np.abs(np.array(wrong.astype(mx.float32)) - right)))})
        return {"versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "torch")},
                "device": "mlx.gpu", "frequencies": native.frequencies,
                "gross_error_bound": 0.02, "cases": rows,
                "dispatch_source": "https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/rope.cpp#L89"}, outputs
    finally:
        mx.set_default_device(original_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700)
    result, outputs = measure()
    path = args.out / "basis.safetensors"
    mx.save_safetensors(str(path), outputs)
    result.update(probe_sha256=sha256(Path(__file__)), output_sha256=sha256(path))
    save(args.out / "basis.json", result)
    check(result)
    print("Eight native-kernel basis cases passed the gross-error check")


if __name__ == "__main__":
    main()
