"""Numerical CUDA gates. No checkpoints or transcript data."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

ATOL = 0.001
RTOL = 1 / 128  # One BF16 epsilon, plus an absolute allowance near zero.
SHAPES = ((2048, 2048, False), (2048, 88064, True), (749, 88813, True), (1, 88814, False))


def require_available(gib):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            value = int(line.split()[1]) * 1024
            if value < gib * 1024**3:
                raise MemoryError(f"Need {gib} GiB MemAvailable, observed {value} bytes")
            return value
    raise ValueError("MemAvailable missing")


def reject_fatal_diagnostics(path):
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            if any(marker in line for marker in ("FATAL:", "CUDA error:", "device-side assert")):
                raise RuntimeError(f"Fatal runtime diagnostic: {line.strip()[:500]}")


def compare(actual, expected):
    import torch
    actual = actual.detach().cpu().to(expected.dtype)
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError("Wrong shape or nonfinite kernel output")
    difference = (actual - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    stats = {"max_abs": difference.max().item(), "rms": difference.square().mean().sqrt().item(),
             "max_tolerance_ratio": (difference / allowed).max().item(),
             "max_relative_where_ref_ge_atol": (difference[expected.abs() >= ATOL] /
                                                 expected.abs()[expected.abs() >= ATOL]).max().item(),
             "outside_tolerance": (difference > allowed).sum().item()}
    if stats["outside_tolerance"]:
        raise ValueError(f"Numerical oracle failed: {stats}")
    return stats


def reference_attention(q, k, v, rows, *, dtype):
    """Compute selected rows head-by-head, without a quadratic CPU score matrix."""
    import torch
    groups = q.shape[1] // k.shape[1]
    offset = k.shape[2] - q.shape[2]
    keys = torch.arange(k.shape[2])
    outputs = []
    for head in range(q.shape[1]):
        scores = q[0, head, rows].to(dtype) @ k[0, head // groups].to(dtype).T
        scores *= q.shape[-1] ** -0.5
        mask = keys[None] <= (torch.tensor(rows)[:, None] + offset)
        scores.masked_fill_(~mask, -torch.inf)
        outputs.append(scores.softmax(-1) @ v[0, head // groups].to(dtype))
    return torch.stack(outputs)[None]


def fixture(q_length, kv_length, generator):
    import torch
    # Match Qwen3's projection -> reshape -> transpose layout.
    q = torch.randn(1, q_length, 16, 128, generator=generator).bfloat16().transpose(1, 2)
    k = torch.randn(1, kv_length, 8, 128, generator=generator).bfloat16().transpose(1, 2)
    v = torch.randn(1, kv_length, 8, 128, generator=generator).bfloat16().transpose(1, 2)
    # Keep long-context outputs away from zero and expose mask/head mapping errors.
    v[..., 0] = 1
    v[..., 1] = torch.linspace(0, 1, kv_length)
    v[..., 2] = 4 * (torch.arange(kv_length) >= kv_length - max(2, q_length // 2))
    v[..., 3] = torch.arange(8)[None, :, None] / 8 + 0.25
    k, v = k.contiguous(), v.contiguous()  # DynamicCache concatenates even its first update.
    return q, k, v


def numeric_worker(stage):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(2)
    if torch.__version__ != "2.14.1+cu130" or torch.version.cuda != "13.0":
        raise ValueError("Expected official Torch 2.14.1+cu130")
    initial_available = require_available(14) if stage == "shapes" else None
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("Expected Spark GB10 capability 12.1")
    allocator_gib = 1 if stage == "tiny" else 3
    torch.cuda.set_per_process_memory_fraction(allocator_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    generator = torch.Generator().manual_seed(17)
    shapes = ((16, 64, True), (1, 64, False)) if stage == "tiny" else SHAPES
    reference_dtype = torch.float64 if stage == "tiny" else torch.float32
    checks = []
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        for q_length, kv_length, masked in shapes:
            q, k, v = fixture(q_length, kv_length, generator)
            rows = sorted(set((0, q_length // 2, q_length - 1)))
            expected = reference_attention(q, k, v, rows, dtype=reference_dtype)
            if expected.abs().max() < 0.01:
                raise ValueError("Degenerate oracle fixture")
            qg, kg, vg = (x.to("cuda") for x in (q, k, v))
            mask = None
            if masked:
                kg, vg = kg.repeat_interleave(2, dim=1), vg.repeat_interleave(2, dim=1)
                mask = torch.arange(kv_length, device="cuda")[None, :] <= (
                    torch.arange(q_length, device="cuda")[:, None] + kv_length - q_length)
                mask = mask[None, None]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = torch.nn.functional.scaled_dot_product_attention(
                    qg, kg, vg, attn_mask=mask, is_causal=not masked and q_length > 1,
                    enable_gqa=not masked)
                torch.cuda.synchronize()
            operators = sorted(event.key for event in profile.key_averages() if "scaled_dot_product" in event.key)
            checks.append({"case": f"attention-{q_length}-{kv_length}", "rows": rows,
                           "q_stride": list(qg.stride()), "k_stride": list(kg.stride()),
                           "operators": operators, **compare(actual[:, :, rows], expected)})
            del qg, kg, vg, actual, mask, q, k, v, expected
            torch.cuda.empty_cache()
            if stage == "shapes":
                checks[-1]["available_bytes_after_case"] = require_available(8)
        if stage == "tiny":
            inputs = torch.randn(16, 128, generator=generator).bfloat16()
            weight = torch.randn(128, 128, generator=generator).bfloat16()
            expected = inputs.double() @ weight.double().T
            actual = torch.nn.functional.linear(inputs.cuda(), weight.cuda())
            torch.cuda.synchronize()
            checks.append({"case": "gemm", **compare(actual, expected)})
    return {"type": "numeric-result", "stage": stage, "seed": 17,
            "initial_available_bytes": initial_available,
            "reference_dtype": str(reference_dtype), "atol": ATOL, "rtol": RTOL,
            "torch": torch.__version__, "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda, "compiled_arches": torch.cuda.get_arch_list(),
            "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "checks": checks}


def read_result(log, stage, exit_code):
    reject_fatal_diagnostics(log)
    if exit_code:
        raise RuntimeError(f"Numeric worker exited {exit_code}; see {log}")
    results = []
    for line in Path(log).read_text().splitlines():
        if line.startswith('{"type": "numeric-result"'):
            results.append(json.loads(line))
    names = {"attention-16-64", "attention-1-64", "gemm"} if stage == "tiny" else {
        f"attention-{q}-{k}" for q, k, _ in SHAPES}
    if len(results) != 1 or {item["case"] for item in results[0]["checks"]} != names:
        raise ValueError("Missing or unexpected numeric worker results")
    if any(item["outside_tolerance"] for item in results[0]["checks"]):
        raise ValueError("Worker reported a numerical failure")
    return {"passed": True, **results[0]}


def run_smoke(stage="tiny", log=Path("/results/cuda-numeric.log")):
    def interrupted(signum, frame):
        raise InterruptedError(f"Numeric smoke received signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with Path(log).open("x") as output:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--stage", stage],
                                    stdout=output, stderr=subprocess.STDOUT, timeout=35 if stage == "tiny" else 110)
        return read_result(log, stage, result.returncode)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", choices=("tiny", "shapes"), default="tiny")
    args = parser.parse_args()
    print(json.dumps(numeric_worker(args.stage) if args.worker else run_smoke(args.stage)))
