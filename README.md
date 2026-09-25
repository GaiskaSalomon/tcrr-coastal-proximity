# Historical variability in tropical-cyclone coastal proximity

Processing and analysis code for an IBTrACS-only study of tropical-cyclone
coastal proximity in the North Atlantic (NA) and Eastern North Pacific (EP),
1949-2025, submitted to *Tropical Cyclone Research and Review* (TCRR, KeAi).

The term "coastal proximity" is intentional: the analysis counts storms whose
active best-track observations approach land within specified distances. It
does not include population, assets, damages or a formal attribution model.

This repository holds the code only. The derived data panels, frozen analysis
outputs, checksums and manuscript source are archived together with this code
in the accompanying Zenodo deposit (see `CITATION.cff` / `.zenodo.json`).

## Contents

- `scripts/`: panel construction, statistical analysis, figures, tables and
  reproducibility auditing.
- `tests/`: unit tests for the external geospatial layer handling.
- `requirements.txt` / `requirements-geospatial.txt`: Python dependencies.
- `LICENSE`, `LICENSES/`: dual license (MIT for code, CC BY 4.0 for
  non-code content in the Zenodo deposit).

## Reproduce from the archived panels

Download the derived panels from the Zenodo deposit into `data/processed/`,
then:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_manifest.py
.venv/bin/python scripts/00_fetch_natural_earth.py
.venv/bin/python scripts/04_statistical_analysis.py
.venv/bin/python scripts/02_make_figures.py
.venv/bin/python scripts/03_make_tables.py
```

Map regeneration additionally requires the geospatial packages and the
Natural Earth files described in the Zenodo deposit's
`data/external/README.md`. If they are absent, a full figure rebuild stops
with an error. An explicitly partial rebuild is available with
`scripts/02_make_figures.py --skip-maps`; it leaves existing maps untouched
and must not be reported as full figure reproduction.

## Citation

See `CITATION.cff`.
