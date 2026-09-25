#!/usr/bin/env python3
"""Verify SHA-256 hashes of the frozen derived panels."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/MANIFEST.sha256"


def main() -> None:
    failures: list[str] = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise SystemExit("Manifest verification failed:\n- " + "\n- ".join(failures))
    print("Manifest verified.")


if __name__ == "__main__":
    main()
