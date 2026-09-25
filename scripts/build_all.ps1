$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source

& $python (Join-Path $scriptDir "01_build_ibtracs_coastal_panel.py")
& $python (Join-Path $scriptDir "04_statistical_analysis.py")
& $python (Join-Path $scriptDir "02_make_figures.py")
& $python (Join-Path $scriptDir "03_make_tables.py")
& (Join-Path $scriptDir "build_pdf.ps1")
