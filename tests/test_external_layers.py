"""Safety and integrity checks for public map-source installation."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/00_fetch_natural_earth.py"
SPEC = importlib.util.spec_from_file_location("map_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(name="layer.shp"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, b"layer content")
    contents = buffer.getvalue()
    return contents, hashlib.sha256(contents).hexdigest()


class MapSourceTests(unittest.TestCase):
    def test_verified_extraction_is_idempotent(self):
        data, digest = payload()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "layer"
            MODULE.unpack_verified(data, digest, target)
            MODULE.unpack_verified(data, digest, target)
            self.assertEqual((target / "layer.shp").read_bytes(), b"layer content")

    def test_bad_hash_does_not_create_destination(self):
        data, _ = payload()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "layer"
            with self.assertRaisesRegex(RuntimeError, "hash"):
                MODULE.unpack_verified(data, "incorrect", target)
            self.assertFalse(target.exists())

    def test_parent_traversal_is_rejected(self):
        data, digest = payload("../outside.shp")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "archive member"):
                MODULE.unpack_verified(data, digest, Path(folder) / "layer")

    def test_local_edits_are_preserved(self):
        data, digest = payload()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            local = target / "layer.shp"
            local.write_bytes(b"local edit")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                MODULE.unpack_verified(data, digest, target)
            self.assertEqual(local.read_bytes(), b"local edit")


if __name__ == "__main__":
    unittest.main()
