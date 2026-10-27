import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from packaging.tags import parse_tag
from wheel.wheelfile import WheelFile

import repair_cusparselt as repair


class RepairTests(unittest.TestCase):
    def fixture(self, directory, metadata=repair.BAD_WHEEL, machine=183):
        source = directory / repair.FILENAME
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = machine.to_bytes(2, "little")
        with WheelFile(source, "w") as wheel:
            wheel.writestr(repair.LIBRARY, bytes(header) + b"synthetic payload")
            wheel.writestr(f"{repair.INFO}/WHEEL", metadata)
            wheel.writestr(f"{repair.INFO}/METADATA", b"Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n")
        return source

    def test_repair_preserves_payload_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with patch.object(repair, "SOURCE_SHA256", repair.digest(source)):
                first = repair.repair(source, root / "first")
                second = repair.repair(source, root / "second")
            self.assertEqual(repair.digest(first), repair.digest(second))
            result = json.loads((first.parent / "repair-result.json").read_text())
            self.assertEqual(result["unchanged_payload_sha256"], {
                repair.LIBRARY: repair.members(source)[repair.LIBRARY],
                f"{repair.INFO}/METADATA": repair.members(source)[f"{repair.INFO}/METADATA"],
            })
            with WheelFile(first) as wheel:
                self.assertEqual(wheel.read(f"{repair.INFO}/WHEEL"), repair.GOOD_WHEEL)

    def test_unknown_source_is_rejected_before_unpack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Unknown source"):
                repair.repair(source, root / "out")
            self.assertFalse((root / "out").exists())

    def test_wrong_tag_and_x86_payload_are_rejected(self):
        for metadata, machine, message in (
            (repair.GOOD_WHEEL, 183, "Unexpected WHEEL"),
            (repair.BAD_WHEEL, 62, "not ELF64"),
        ):
            with self.subTest(machine=machine), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.fixture(root, metadata, machine)
                with patch.object(repair, "SOURCE_SHA256", repair.digest(source)):
                    with self.assertRaisesRegex(ValueError, message):
                        repair.repair(source, root / "out")

    def test_repaired_tag_matches_arm64(self):
        supported = parse_tag("py3-none-manylinux2014_aarch64")
        self.assertTrue(parse_tag("py3-none-manylinux2014_sbsa").isdisjoint(supported))
        self.assertFalse(parse_tag("py3-none-manylinux2014_aarch64").isdisjoint(supported))

    def test_record_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with zipfile.ZipFile(source) as wheel:
                contents = {name: wheel.read(name) for name in wheel.namelist()}
            contents[repair.LIBRARY] += b"corruption"
            with zipfile.ZipFile(source, "w") as wheel:
                for name, payload in contents.items():
                    wheel.writestr(name, payload)
            with patch.object(repair, "SOURCE_SHA256", repair.digest(source)):
                with self.assertRaisesRegex(Exception, "Hash mismatch"):
                    repair.repair(source, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
