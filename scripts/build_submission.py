#!/usr/bin/env python3
"""Build the Tropical Cyclone Research and Review submission package.

TCRR asks for illustrations as separate files with their captions supplied
separately rather than attached to the figure, and for a single-column layout
with author--year citations. The master manuscript keeps figures in place for
reading, so this script inlines the section files, replaces each figure with a
placement marker, and gathers the captions into a list at the end. Tables stay
where they are: the separate-caption rule is stated for illustrations.

The previous target was Natural Hazards, whose package was a plain copy of the
master. That layout is no longer produced.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission/tcrr"

INPUT_RE = re.compile(r"^[ \t]*\\input\{sections/([^}]+)\}[ \t]*$", re.MULTILINE)
FIGURE_RE = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL)
GRAPHIC_RE = re.compile(r"^[ \t]*\\includegraphics(?:\[[^]]*\])?\{[^}]+\}[ \t]*$",
                        re.MULTILINE)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def inline_sections(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        path = ROOT / "pdf/sections" / f"{name}.tex"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    return INPUT_RE.sub(replace, source)


def separate_captions(source: str) -> str:
    """Replace each figure with a marker and list the captions at the end."""
    figures = FIGURE_RE.findall(source)
    if not figures:
        raise RuntimeError("No figure environments found; check the master layout.")

    def marker(match: re.Match[str]) -> str:
        label = LABEL_RE.search(match.group(0))
        if label is None:
            raise RuntimeError("A submission figure has no label.")
        return ("\n\\begin{center}[Figure~\\ref{" + label.group(1)
                + "} about here]\\end{center}\n")

    transformed = FIGURE_RE.sub(marker, source)
    if not transformed.rstrip().endswith(r"\end{document}"):
        raise RuntimeError("Unexpected manuscript ending.")
    transformed = transformed.rsplit(r"\end{document}", 1)[0]

    captions = [r"\clearpage", r"\section*{Figure captions}"]
    for figure in figures:
        caption_only = GRAPHIC_RE.sub("", figure)
        caption_only = re.sub(r"\\begin\{figure\*?\}(\[[^]]*\])?",
                              r"\\begin{figure}[h!]", caption_only)
        caption_only = caption_only.replace(r"\end{figure*}", r"\end{figure}")
        captions.append(caption_only)

    return (transformed.rstrip() + "\n\n" + "\n\n".join(captions)
            + "\n\n\\end{document}\n")


def copy_matching(source: Path, destination: Path, pattern: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob(pattern))
    for path in paths:
        shutil.copy2(path, destination / path.name)
    return len(paths)


def write_zip(destination: Path, base: Path, paths: list[Path]) -> None:
    """Write a deterministic, portable ZIP from paths below ``base``."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths, key=lambda item: item.as_posix().lower()):
            if path.is_dir():
                members = sorted(path.rglob("*"))
            else:
                members = [path]
            for member in members:
                if not member.is_file():
                    continue
                if "__pycache__" in member.parts or member.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(member, member.relative_to(base).as_posix())


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    master = (ROOT / "pdf/main.tex").read_text(encoding="utf-8")
    (OUTPUT / "main.tex").write_text(
        separate_captions(inline_sections(master)), encoding="utf-8")

    supplement_master = (ROOT / "pdf/supplement.tex").read_text(encoding="utf-8")
    (OUTPUT / "supplement.tex").write_text(
        inline_sections(supplement_master), encoding="utf-8")
    shutil.copy2(ROOT / "pdf/references.bib", OUTPUT / "references.bib")

    # Cada ilustracion existe en PDF y en PNG, y el manuscrito usa una sola. Se
    # copia exactamente lo que el texto referencia: subir las dos versiones
    # entregaria dieciocho archivos para nueve figuras.
    referenced = sorted(set(re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
        inline_sections(master))))
    figures_out = OUTPUT / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)
    for name in referenced:
        source = ROOT / "pdf/figures" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, figures_out / source.name)
    n_figures = len(referenced)

    copy_matching(ROOT / "pdf/tables", OUTPUT / "tables", "*.tex")

    for name in (
        "cover_letter.txt",
        "SUBMISSION_METADATA.md",
        "declaration_of_competing_interests.docx",
    ):
        source = ROOT / "pdf" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUTPUT / name)

    (OUTPUT / "README.txt").write_text(
        "Tropical Cyclone Research and Review submission package.\n\n"
        "main.tex is self-contained: the section files are inlined and each\n"
        "figure is replaced by a placement marker, with the captions gathered\n"
        "under 'Figure captions' at the end, as the journal requires captions\n"
        "supplied separately from the illustrations.\n\n"
        f"Upload each of the {n_figures} files under figures/ as a separate\n"
        "illustration. Vector figures are PDF; raster figures are at or above\n"
        "300 dpi.\n\n"
        "The layout is single column with author-year citations; references are\n"
        "sorted alphabetically and then chronologically using Elsevier's\n"
        "elsarticle-harv style. Acknowledgements and the declaration sections\n"
        "appear before the references.\n\n"
        "supplement.tex is the Online Supplementary Material and compiles\n"
        "separately. The separate DOCX declaration of competing interests is\n"
        "also included and must remain in DOCX format.\n\n"
        "Compile with: latexmk -pdf -interaction=nonstopmode -halt-on-error\n",
        encoding="utf-8",
    )

    source_members = [
        OUTPUT / "main.tex",
        OUTPUT / "supplement.tex",
        OUTPUT / "references.bib",
        OUTPUT / "README.txt",
        OUTPUT / "figures",
        OUTPUT / "tables",
    ]
    write_zip(OUTPUT / "A3-TCRR-LaTeX-source.zip", OUTPUT, source_members)

    reproducibility_members = [
        ROOT / ".zenodo.json",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "LICENSES",
        ROOT / "Makefile",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-geospatial.txt",
        ROOT / "data/processed",
        ROOT / "data/external/README.md",
        ROOT / "outputs",
        ROOT / "scripts",
        ROOT / "supplement",
        ROOT / "tests",
        ROOT / "pdf/main.tex",
        ROOT / "pdf/supplement.tex",
        ROOT / "pdf/references.bib",
        ROOT / "pdf/sections",
        ROOT / "pdf/tables",
        ROOT / "pdf/figures",
    ]
    write_zip(OUTPUT / "A3-TCRR-reproducibility.zip", ROOT, reproducibility_members)
    print(f"Built TCRR package with {n_figures} separate illustrations.")


if __name__ == "__main__":
    main()
