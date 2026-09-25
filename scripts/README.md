# Scripts

Run scripts from the repository root unless noted otherwise.

## Workflow

1. Build the North Atlantic (NA) and Eastern North Pacific (EP) storm-level,
   observation-level, and annual panels.
2. Run trend, bootstrap, Wasserstein, Monte Carlo power, and GLM diagnostics.
3. Generate article figures.
4. Generate manuscript-ready tables.
5. Compile the manuscript in `pdf/`.

## Commands from versioned derived panels (Linux environment)

```bash
./.venv/bin/python scripts/04_statistical_analysis.py
./.venv/bin/python scripts/02_make_figures.py
./.venv/bin/python scripts/03_make_tables.py
```

Map figures additionally require geopandas, pyproj and the Natural Earth files
documented in the article README. The build-panel step additionally requires the
uncommitted IBTrACS NetCDF. Without these inputs, the figure script skips maps
and regenerates the non-map figures.

If only the frozen map PDFs are available, this compatibility step converts
their legacy Type-3 text to vector outlines for journal submission:

```bash
python scripts/05_outline_legacy_map_fonts.py
```

## Windows commands

```powershell
python .\scripts\01_build_ibtracs_coastal_panel.py
python .\scripts\04_statistical_analysis.py
python .\scripts\02_make_figures.py
python .\scripts\03_make_tables.py
.\scripts\build_pdf.ps1
```

The default statistical run uses 20,000 primary moving-block bootstrap and
20,000 Monte Carlo replicates, plus 500 bootstrap replicates for the
Wasserstein and block-length robustness checks. For
quick testing, reduce them:

```powershell
python .\scripts\04_statistical_analysis.py --bootstrap-reps 200 --mc-reps 200 --sensitivity-reps 100
```

## Files

- `01_build_ibtracs_coastal_panel.py` -- reads the IBTrACS NetCDF and builds
  storm, observation, and annual panels for 1949--2025.
- `04_statistical_analysis.py` -- Mann-Kendall/Sen, moving-block bootstrap,
  Wasserstein distances, Monte Carlo power, binomial GLM diagnostics,
  Holm/BH-FDR multiplicity adjustments, observational-era sensitivity, and
  block-length sensitivity.
- `02_make_figures.py` -- figure-generation script.
- `03_make_tables.py` -- table-generation script.
- `05_outline_legacy_map_fonts.py` -- removes Type-3 glyphs from frozen maps
  when their unversioned cartographic sources are unavailable.
- `build_pdf.ps1` -- local LaTeX build helper.
