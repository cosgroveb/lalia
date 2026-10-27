"""Repair one known NVIDIA wheel tag; preserve and verify every payload member."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from wheel.wheelfile import WheelFile


FILENAME = "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"
SOURCE_SHA256 = "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
INFO = "nvidia_cusparselt_cu13-0.8.1.dist-info"
BAD_WHEEL = (
    b"Wheel-Version: 1.0\nGenerator: setuptools (80.9.0)\n"
    b"Root-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n\n"
)
GOOD_WHEEL = BAD_WHEEL.replace(b"manylinux2014_sbsa", b"manylinux2014_aarch64")
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def members(path):
    # WheelFile verifies each member against RECORD while reading it.
    with WheelFile(path) as wheel:
        hashes = {}
        for name in wheel.namelist():
            with wheel.open(name) as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return hashes


def repair(source, destination):
    if source.name != FILENAME or digest(source) != SOURCE_SHA256:
        raise ValueError("Unknown source wheel; refusing repair")
    original = members(source)
    destination.mkdir()
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(source), "-d", temporary],
            check=True,
        )
        unpacked = Path(temporary) / "nvidia_cusparselt_cu13-0.8.1"
        metadata = unpacked / INFO / "WHEEL"
        if metadata.read_bytes() != BAD_WHEEL:
            raise ValueError("Unexpected WHEEL metadata")
        with (unpacked / LIBRARY).open("rb") as stream:
            header = stream.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xb7\x00":
            raise ValueError("Payload is not ELF64 little-endian AArch64")
        metadata.write_bytes(GOOD_WHEEL)
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(destination)],
            env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"},
            check=True,
        )
    repaired = destination / FILENAME
    updated = members(repaired)
    if original.keys() != updated.keys():
        raise ValueError("Repair changed archive membership")
    changed = {name for name in original if original[name] != updated[name]}
    if changed != {f"{INFO}/WHEEL", f"{INFO}/RECORD"}:
        raise ValueError(f"Repair changed payload: {sorted(changed)}")
    with WheelFile(repaired) as wheel:
        if wheel.read(f"{INFO}/WHEEL") != GOOD_WHEEL:
            raise ValueError("Repacked platform tag is incorrect")
    result = {
        "source_sha256": SOURCE_SHA256,
        "repaired_sha256": digest(repaired),
        "changed_members": sorted(changed),
        "unchanged_payload_sha256": {name: original[name] for name in original if name not in changed},
    }
    (destination / "repair-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return repaired


if __name__ == "__main__":
    repair(Path(sys.argv[1]), Path(sys.argv[2]))
