#!/usr/bin/env python3
"""Fail if the manuscript drifts from the frozen analysis outputs.

The panel counts and the headline test results reach the manuscript as literal
text. Nothing but this script keeps them tied to the JSON and CSV that the
analysis wrote, so a rerun that shifted a count, or an edit that mistyped one,
would otherwise pass unnoticed until review.

Also verifies that every file under ``outputs/`` is covered by a SHA-256
manifest, which the package previously applied only to the derived panels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SECTIONS = ROOT / "pdf/sections"
MANIFEST = OUTPUTS / "FROZEN.sha256"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def manuscript_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(SECTIONS.glob("*.tex"))]
    parts.append((ROOT / "pdf/main.tex").read_text(encoding="utf-8"))
    return "\n".join(parts)


def grouped(value: int) -> list[str]:
    """The manuscript writes thousands with a comma or a LaTeX thin space."""
    plain = f"{value:d}"
    comma = f"{value:,}"
    return [plain, comma, comma.replace(",", "{,}"), comma.replace(",", "\\,")]


def main() -> None:
    panel = json.loads((OUTPUTS / "panel_summary.json").read_text(encoding="utf-8"))
    text = manuscript_text()

    checks = {
        "storm count": panel["n_storms"],
        "observation count": panel["n_observations"],
        "NA basin count": panel["basin_counts"]["NA"],
        "EP basin count": panel["basin_counts"]["EP"],
        "NA 100-km approach count": panel["approach_100km_counts"]["NA"],
        "EP 100-km approach count": panel["approach_100km_counts"]["EP"],
    }
    for label, value in checks.items():
        require(any(form in text for form in grouped(value)),
                f"Manuscript does not track the frozen {label}: {value}")

    require(str(panel["season_min"]) in text and str(panel["season_max"]) in text,
            "Manuscript does not state the frozen analysis window.")

    # Las pruebas titulares deben seguir existiendo y no haber cambiado de forma.
    wasserstein = pd.read_csv(OUTPUTS / "wasserstein_results.csv", keep_default_na=False)
    require(not wasserstein.empty, "The Wasserstein results table is empty.")
    require({"basin", "metric", "wasserstein"} <= set(wasserstein.columns),
            "The Wasserstein results table lost a required column.")
    # "NA" es el codigo del Atlantico norte, no un valor faltante.
    require(set(wasserstein["basin"]) == {"NA", "EP"},
            f"Basin codes changed: {sorted(set(wasserstein['basin']))}")

    summary = json.loads(
        (OUTPUTS / "statistical_analysis_summary.json").read_text(encoding="utf-8"))
    for field in ("bootstrap_reps", "monte_carlo_reps"):
        require(any(form in text for form in grouped(summary[field])),
                f"Manuscript does not state the frozen {field}: {summary[field]}")

    # --- integridad de outputs/ --------------------------------------------
    require(MANIFEST.is_file(),
            "outputs/FROZEN.sha256 is missing; generate it with --write.")
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.lstrip("* ").strip()] = digest
    present = {
        p.name
        for p in OUTPUTS.iterdir()
        if p.is_file() and p.name != MANIFEST.name and not p.name.startswith(".")
    }
    require(present == set(listed),
            f"Manifest and outputs/ disagree: {present ^ set(listed)}")
    for name, digest in listed.items():
        actual = hashlib.sha256((OUTPUTS / name).read_bytes()).hexdigest()
        require(actual == digest, f"Checksum mismatch in outputs/{name}")

    print(f"Frozen-result audit passed: panel counts, headline tables and "
          f"{len(listed)} output files are traceable.")


def write_manifest() -> None:
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted(OUTPUTS.iterdir())
             if p.is_file() and p.name != MANIFEST.name and not p.name.startswith(".")]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote manifest for {len(lines)} output files.")


if __name__ == "__main__":
    import sys
    write_manifest() if "--write" in sys.argv else main()
