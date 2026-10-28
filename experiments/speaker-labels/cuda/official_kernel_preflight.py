"""Exercise the official frontend and Qwen outer-product kernels without a model."""
import json
import os
from pathlib import Path

from official_transcribe import sha256


def main():
    import torch

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GB10":
        raise ValueError("Kernel preflight requires NVIDIA GB10")
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    real = torch.linspace(-1, 1, 16)
    spectrum = torch.complex(real, real.flip(0))
    magnitudes = spectrum.cuda().abs() ** 2
    torch.testing.assert_close(magnitudes.cpu(), spectrum.abs() ** 2, rtol=1e-6, atol=1e-7)

    inv_freq = (1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)))[None, :, None]
    positions = torch.arange(16, dtype=torch.float32)[None, None, :]
    actual = (inv_freq.cuda() @ positions.cuda()).transpose(1, 2)
    expected = (inv_freq @ positions).transpose(1, 2)
    torch.cuda.synchronize()
    if not torch.isfinite(actual).all() or not torch.isfinite(magnitudes).all():
        raise ValueError("Nonfinite kernel output")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    caches = {}
    for variable, location in (("TRITON_CACHE_DIR", "/tmp/triton"),
                               ("PYTORCH_KERNEL_CACHE_PATH", "/tmp/torch-kernels")):
        if os.environ.get(variable) != location:
            raise ValueError(f"Unexpected {variable}")
        paths = sorted(path for path in Path(location).rglob("*") if path.is_file())
        if not paths:
            raise ValueError(f"Kernel did not populate {location}")
        caches[variable] = {"path": location, "files": {
            str(path.relative_to(location)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths}}
    if not any("bmm_outer_product" in name and name.endswith(".cubin")
               for name in caches["TRITON_CACHE_DIR"]["files"]):
        raise ValueError("Qwen-shaped matmul did not compile the expected Triton kernel")
    print(json.dumps({"status": "ready", "image": os.environ["FRONTEND_IMAGE_ID"],
                      "source_inventory_sha256": os.environ["SOURCE_INVENTORY_SHA256"],
                      "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                      "outer_product_shape": list(actual.shape), "outer_product_max_error": 0.0,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "root_read_only": bool(os.statvfs("/").f_flag & os.ST_RDONLY),
                      "tmp_mount": next(line for line in Path("/proc/self/mountinfo").read_text().splitlines()
                                        if " /tmp " in line), "caches": caches}))


if __name__ == "__main__":
    main()
