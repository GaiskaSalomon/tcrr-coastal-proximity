$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pdf = Join-Path $root "pdf"

Push-Location $pdf
try {
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
}
finally {
    Pop-Location
}

