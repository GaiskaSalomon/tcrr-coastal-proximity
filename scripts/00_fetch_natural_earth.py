"""Fetch the three pinned public-domain Natural Earth map layers."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile


ARTICLE = Path(__file__).resolve().parents[1]
LAYERS = (
    ("countries_50m", "50m_cultural/ne_50m_admin_0_countries.zip",
     "5fed433373581fa648920435f937d95f2d3c0200e067409c6478dcdf1b853139"),
    ("coastline_50m", "50m_physical/ne_50m_coastline.zip",
     "640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde"),
    ("ocean_50m", "50m_physical/ne_50m_ocean.zip",
     "abf268ba229f5eaab5012d7bfba340b424a4d25ed1efa085815c4a17ee0fd77d"),
)


def unpack_verified(payload: bytes, digest: str, destination: Path) -> None:
    if hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError("Natural Earth archive hash differs from the pinned source; no extraction.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # These provider archives are flat. Reject unsafe or unexpected paths
        # and inspect existing content before writing any of the layer files.
        pending = []
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or len(name.parts) != 1 or name.name in {".", ".."}:
                raise RuntimeError(f"Unexpected archive member: {entry.filename}")
            if entry.is_dir():
                continue
            target = destination / name.name
            contents = archive.read(entry)
            if target.is_symlink() or (target.exists() and target.read_bytes() != contents):
                raise RuntimeError(f"Existing map source differs; refusing to overwrite {target}")
            pending.append((target, contents))
        destination.mkdir(parents=True, exist_ok=True)
        for target, contents in pending:
            if not target.exists():
                target.write_bytes(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=ARTICLE / "data/external/natural_earth")
    args = parser.parse_args()
    for folder, resource, digest in LAYERS:
        url = "https://naturalearth.s3.amazonaws.com/" + resource
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read(50_000_001)
        if len(payload) > 50_000_000:
            raise RuntimeError("Unexpectedly large map archive; no extraction.")
        unpack_verified(payload, digest, args.destination / folder)
        print(f"Verified and installed {folder}: {digest}", flush=True)


if __name__ == "__main__":
    main()
