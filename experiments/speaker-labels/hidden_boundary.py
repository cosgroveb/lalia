"""Capture the final hidden boundary and compare the unchanged native heads."""

import hashlib
import json
from pathlib import Path

WIDTH, VOCAB = 1024, 151936
WEIGHT_KEY = "model.language_model.embed_tokens.weight"
WEIGHT_SHA = "201a1b803288a3775e7367ba87ccc47f0a919685643729eca438195a5e890dc2"
RAW_SHA = {"mac": "ac43a5b368d5b3e61eed4a8f89d7373571b283caeab9eebc6003f64a1713d8fc",
           "cuda": "0c9d495eff6e700b42d28b9abe8f17969297013d63bd70324c3eb230e615c7d7"}
SHAPES = {"mac": (1, 1, WIDTH), "cuda": (1, WIDTH)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def require_capture(report, backend):
    if (report["backend"] != backend or report["hidden_shape"] != list(SHAPES[backend])
            or report["dtype"] != "BF16" or report["weight_shape"] != [VOCAB, WIDTH]
            or report["weight_sha256"] != WEIGHT_SHA or report["logits_sha256"] != RAW_SHA[backend]):
        raise ValueError("Hidden capture did not reproduce the pinned native head")


def capture_receipt(directory, backend):
    report = json.loads((directory / f"{backend}-head.json").read_text())
    require_capture(report, backend)
    if sha256(directory / f"{backend}-head.safetensors") != report["file_sha256"]:
        raise ValueError("Hidden capture file changed")
    return report


def mlx_digest(value):
    import mlx.core as mx
    import numpy as np

    if value.dtype != mx.bfloat16 or not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError("Require finite BF16 boundary tensors")
    return hashlib.sha256(np.array(value.view(mx.uint16)).tobytes()).hexdigest()


def torch_digest(value):
    import torch

    if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
        raise ValueError("Require finite BF16 boundary tensors")
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def require_tensor(digest, value, shape, expected=None):
    if tuple(value.shape) != tuple(shape):
        raise ValueError("Boundary tensor shape differs")
    actual = digest(value)
    if expected is not None and actual != expected:
        raise ValueError("Boundary tensor bytes differ")
    return actual


def capture_mlx(hidden, logits, embedding, directory):
    import mlx.core as mx
    import mlx.nn as nn

    if type(embedding) is not nn.Embedding:
        raise ValueError("Require the native tied Embedding head")
    # The caller has already evaluated and gated the unchanged full projection.
    report = {"backend": "mac", "dtype": "BF16", "hidden_shape": list(hidden.shape),
              "hidden_sha256": require_tensor(mlx_digest, hidden, SHAPES["mac"]),
              "logits_sha256": require_tensor(mlx_digest, logits, (VOCAB,), RAW_SHA["mac"]),
              "weight_shape": list(embedding.weight.shape),
              "weight_sha256": require_tensor(mlx_digest, embedding.weight, (VOCAB, WIDTH), WEIGHT_SHA),
              "layout": "MLX array shape recorded; stride metadata is not exposed"}
    path = directory / "mac-head.safetensors"
    mx.save_safetensors(str(path), {"hidden": hidden, "logits": logits})
    report["file_sha256"] = sha256(path)
    require_capture(report, "mac")
    save(directory / "mac-head.json", report)
    return report


def read_mlx(directory, backend):
    import mlx.core as mx

    report = capture_receipt(directory, backend)
    values = mx.load(str(directory / f"{backend}-head.safetensors"))
    require_tensor(mlx_digest, values["hidden"], SHAPES[backend], report["hidden_sha256"])
    require_tensor(mlx_digest, values["logits"], (VOCAB,), RAW_SHA[backend])
    return values, report


def read_torch(directory, backend, device="cpu"):
    from safetensors.torch import load_file

    report = capture_receipt(directory, backend)
    values = load_file(directory / f"{backend}-head.safetensors", device=device)
    require_tensor(torch_digest, values["hidden"], SHAPES[backend], report["hidden_sha256"])
    require_tensor(torch_digest, values["logits"], (VOCAB,), RAW_SHA[backend])
    return values, report


def torch_capture_heads(hidden, logits, weight, directory, mac_directory):
    import torch
    from safetensors.torch import save_file

    report = {"backend": "cuda", "dtype": "BF16", "hidden_shape": list(hidden.shape),
              "hidden_sha256": require_tensor(torch_digest, hidden, SHAPES["cuda"]),
              "logits_sha256": require_tensor(torch_digest, logits, (VOCAB,), RAW_SHA["cuda"]),
              "weight_shape": list(weight.shape),
              "weight_sha256": require_tensor(torch_digest, weight, (VOCAB, WIDTH), WEIGHT_SHA),
              "hidden_stride": list(hidden.stride()), "weight_stride": list(weight.stride())}
    path = directory / "cuda-head.safetensors"
    save_file({"hidden": hidden.detach().cpu().contiguous(),
               "logits": logits.detach().cpu().contiguous()}, path)
    report["file_sha256"] = sha256(path)
    require_capture(report, "cuda")
    save(directory / "cuda-head.json", report)
    own, _ = read_torch(directory, "cuda", str(weight.device))
    other, mac = read_torch(mac_directory, "mac", str(weight.device))
    if report["hidden_sha256"] == mac["hidden_sha256"]:
        return {"identical_hidden": True, "capture": report}
    own_hidden = own["hidden"].reshape(SHAPES["cuda"])
    other_hidden = other["hidden"].reshape(SHAPES["cuda"])
    if tuple(own_hidden.stride()) != tuple(hidden.stride()) or tuple(other_hidden.stride()) != tuple(hidden.stride()):
        raise ValueError("CUDA serialized hidden stride differs from native projection")
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        self_logits = torch.nn.functional.linear(own_hidden, weight)[0]
        require_tensor(torch_digest, self_logits, (VOCAB,), RAW_SHA["cuda"])
        cross_logits = torch.nn.functional.linear(other_hidden, weight)[0]
    require_tensor(torch_digest, cross_logits, (VOCAB,))
    output = directory / "head-projections.safetensors"
    save_file({"self_logits": self_logits.cpu().contiguous(),
               "cross_logits": cross_logits.cpu().contiguous()}, output)
    return {"identical_hidden": False, "capture": report, "self_sha256": torch_digest(self_logits),
            "cross_sha256": torch_digest(cross_logits), "self_shape": list(own_hidden.shape),
            "cross_shape": list(other_hidden.shape), "cross_stride": list(other_hidden.stride()),
            "projections_sha256": sha256(output)}


def mlx_heads(model_path, mac_directory, cuda_directory, directory):
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from safetensors import safe_open

    own, mac = read_mlx(mac_directory, "mac")
    other, cuda = read_mlx(cuda_directory, "cuda")
    if mac["hidden_sha256"] == cuda["hidden_sha256"]:
        raise ValueError("Identical hidden states already locate the head-path difference")
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        saved_weight = checkpoint.get_tensor(WEIGHT_KEY)
    require_tensor(torch_digest, saved_weight, (VOCAB, WIDTH), WEIGHT_SHA)
    # The selective MLX loader cannot read BF16 in this pinned safetensors build.
    weight = mx.array(saved_weight.view(torch.uint16).numpy()).view(mx.bfloat16)
    mx.eval(weight)
    del saved_weight
    require_tensor(mlx_digest, weight, (VOCAB, WIDTH), WEIGHT_SHA)
    # Avoid allocating a random full embedding matrix before loading its bytes.
    embedding = nn.Embedding.__new__(nn.Embedding)
    nn.Module.__init__(embedding)
    embedding.weight = weight
    self_hidden = own["hidden"].reshape(SHAPES["mac"])
    cross_hidden = other["hidden"].reshape(SHAPES["mac"])
    self_logits = embedding.as_linear(self_hidden)[0, 0]
    require_tensor(mlx_digest, self_logits, (VOCAB,), RAW_SHA["mac"])
    cross_logits = embedding.as_linear(cross_hidden)[0, 0]
    require_tensor(mlx_digest, cross_logits, (VOCAB,))
    output = directory / "scores.safetensors"
    mx.save_safetensors(str(output), {"logits": self_logits, "cross_logits": cross_logits})
    return {"status": "complete", "self_sha256": mlx_digest(self_logits),
            "cross_sha256": mlx_digest(cross_logits), "weight_sha256": WEIGHT_SHA,
            "self_shape": list(self_hidden.shape), "cross_shape": list(cross_hidden.shape),
            "scores_sha256": sha256(output)}


def analyze(mac_directory, cuda_directory, head_directory):
    """CPU-only analysis, with one FP64 normalization applied to all raw vectors."""
    import numpy as np
    from safetensors.torch import load_file

    mac, mac_receipt = read_torch(mac_directory, "mac")
    cuda, cuda_receipt = read_torch(cuda_directory, "cuda")
    if mac_receipt["hidden_sha256"] == cuda_receipt["hidden_sha256"]:
        return {"identical_hidden": True, "interpretation": "Different native heads on identical hidden bytes"}
    cuda_result = json.loads((cuda_directory / "result.json").read_text())["boundary"]
    mac_result = json.loads((head_directory / "result.json").read_text())
    if mac_result["status"] != "complete" or cuda_result["identical_hidden"]:
        raise ValueError("Head comparison did not complete")
    for path, expected in ((cuda_directory / "head-projections.safetensors", cuda_result["projections_sha256"]),
                           (head_directory / "scores.safetensors", mac_result["scores_sha256"])):
        if sha256(path) != expected:
            raise ValueError("Head projection file changed")
    cuda_heads = load_file(cuda_directory / "head-projections.safetensors", device="cpu")
    mac_heads = load_file(head_directory / "scores.safetensors", device="cpu")
    vectors = {"mac_head_mac_hidden": mac_heads["logits"], "mac_head_cuda_hidden": mac_heads["cross_logits"],
               "cuda_head_mac_hidden": cuda_heads["cross_logits"], "cuda_head_cuda_hidden": cuda_heads["self_logits"]}
    expected = (RAW_SHA["mac"], mac_result["cross_sha256"], cuda_result["cross_sha256"], RAW_SHA["cuda"])
    for (name, values), digest in zip(vectors.items(), expected):
        require_tensor(torch_digest, values, (VOCAB,), digest)
    arrays = {name: values.float().numpy().astype(np.float64) for name, values in vectors.items()}

    def difference(left, right):
        values = left - right
        return {"different_elements": int(np.count_nonzero(values)),
                "max_absolute_difference": float(np.abs(values).max()),
                "mean_absolute_difference": float(np.abs(values).mean())}

    table = {}
    for name, values in arrays.items():
        continuation = values.copy()
        continuation[[151643, 151645]] = -np.inf
        best = int(continuation.argmax())
        normalized = values - (values.max() + np.log(np.exp(values - values.max()).sum()))
        table[name] = {"raw_bf16_sha256": torch_digest(vectors[name]), "argmax": int(values.argmax()),
                       "tie_count": int(np.count_nonzero(values == values.max())), "eos": float(values[151645]),
                       "token_3821": float(values[3821]), "best_continuation_id": best,
                       "eos_margin": float(values[151645] - values[best]),
                       "analysis_fp64_logprob_eos": float(normalized[151645]),
                       "analysis_fp64_logprob_3821": float(normalized[3821])}
    mm, mc, cm, cc = (arrays[name] for name in vectors)
    return {"identical_hidden": False, "table": table,
            "normalization": "Identical CPU FP64 logsumexp analysis, not native generation probabilities",
            "hidden_difference": difference(mac["hidden"].float().numpy().reshape(-1),
                                            cuda["hidden"].float().numpy().reshape(-1)),
            "observed_logits_difference": difference(cc, mm),
            "anchored_differences": {"hidden_at_mac_head": difference(mc, mm),
                                      "head_at_cuda_hidden": difference(cc, mc),
                                      "head_at_mac_hidden": difference(cm, mm),
                                      "hidden_at_cuda_head": difference(cc, cm)}}
