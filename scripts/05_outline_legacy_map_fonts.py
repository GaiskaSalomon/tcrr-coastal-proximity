#!/usr/bin/env python3
"""Convert fonts in frozen map PDFs to vector outlines.

The cartographic source inputs are intentionally outside the versioned article
package. When those inputs are unavailable, this post-processing step removes
legacy Type-3 glyphs from the already versioned map PDFs without changing the
underlying plotted data. A full source rebuild with ``02_make_figures.py`` uses
TrueType embedding directly and does not require this compatibility step.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess


ARTICLE = Path(__file__).resolve().parents[1]
FIGURES = ARTICLE / "pdf" / "figures"
MAPS = [
    "fig01_tracks_lmi.pdf",
    "fig06_coastal_density_map.pdf",
    "fig07_regional_coastal_approaches.pdf",
    "fig08_return_period_map.pdf",
    "fig09_early_recent_density.pdf",
]


def has_type3_fonts(path: Path) -> bool:
    """Return whether Poppler reports any Type-3 font in *path*.

    If ``pdffonts`` is unavailable, process the file conservatively rather than
    claiming that it has already passed the compatibility check.
    """
    pdffonts = shutil.which("pdffonts")
    if pdffonts is None:
        return True
    result = subprocess.run(
        [pdffonts, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return "Type 3" in result.stdout


def main() -> None:
    gs = shutil.which("gs")
    if gs is None:
        raise RuntimeError("Ghostscript (gs) is required to outline frozen map fonts")

    for name in MAPS:
        source = FIGURES / name
        if not source.exists():
            raise FileNotFoundError(source)
        if not has_type3_fonts(source):
            print("already compatible", source)
            continue
        temporary = source.with_name(f".{source.stem}.outlined.pdf")
        subprocess.run(
            [
                gs,
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dNoOutputFonts",
                f"-sOutputFile={temporary}",
                str(source),
            ],
            check=True,
        )
        os.replace(temporary, source)
        print("outlined", source)


if __name__ == "__main__":
    main()
