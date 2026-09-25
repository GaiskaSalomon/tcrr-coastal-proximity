#!/usr/bin/env python3
"""Mark the no-competing-interests option in Elsevier's official DOCX template."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.template.is_file():
        raise FileNotFoundError(args.template)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    replaced = False
    with zipfile.ZipFile(args.template, "r") as source, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "word/document.xml":
                text = data.decode("utf-8")

                # Keep Word's canonical checkbox state and the visible glyphs in
                # sync.  The first declaration is selected; the remaining two
                # are explicitly cleared.
                checkbox_index = 0

                def set_checkbox_state(match: re.Match[str]) -> str:
                    nonlocal checkbox_index
                    value = "1" if checkbox_index == 0 else "0"
                    checkbox_index += 1
                    return f'{match.group(1)}{value}{match.group(2)}'

                text = re.sub(
                    r'(<w14:checked\s+w14:val=")[01]("/>)',
                    set_checkbox_state,
                    text,
                )
                if checkbox_index != 3:
                    raise RuntimeError(
                        f"Expected 3 checkbox controls, found {checkbox_index}."
                    )

                glyph_index = 0

                def set_checkbox_glyph(match: re.Match[str]) -> str:
                    nonlocal glyph_index
                    glyph = "☒" if glyph_index == 0 else "☐"
                    glyph_index += 1
                    return f"{match.group(1)}{glyph}{match.group(3)}"

                text = re.sub(
                    r"(<w:t[^>]*>)([☐☒])(</w:t>)",
                    set_checkbox_glyph,
                    text,
                )
                if glyph_index != 3:
                    raise RuntimeError(
                        f"Expected 3 visible checkbox glyphs, found {glyph_index}."
                    )
                data = text.encode("utf-8")
                replaced = True
            destination.writestr(item, data)

    if not replaced:
        raise RuntimeError("The document body was not found in the template.")
    print(f"Prepared {args.output}")


if __name__ == "__main__":
    main()
