"""Generate manuscript tables from processed IBTrACS panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARTICLE = Path(__file__).resolve().parents[1]
PROCESSED = ARTICLE / "data" / "processed"
OUTPUTS = ARTICLE / "outputs"
TABLE_DIR = ARTICLE / "pdf" / "tables"

LABELS = {"NA": "NA", "EP": "EP"}


def fmt(x: object, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{digits}f}"


def fmt_p(x: object) -> str:
    if pd.isna(x):
        return "--"
    value = float(x)
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_summary(storms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basin, sub in storms.groupby("basin"):
        rows.append(
            {
                "Basin": LABELS[basin],
                "Storms": len(sub),
                "Hurricanes": int((sub["vmax_lifetime"] >= 64).sum()),
                "Major hurricanes": int((sub["vmax_lifetime"] >= 96).sum()),
                "Landfalls": int(sub["landfall_ibtracs"].sum()),
                "Within 100 km": int(sub["approach_100km"].sum()),
                "Mean LMI lat": sub["phi_lmi"].mean(),
                "Median dmin": sub["dmin_coast_km"].median(),
            }
        )
    return pd.DataFrame(rows)


def latex_sample_summary(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{IBTrACS sample summary, 1949--2025.}",
        "\\label{tab:sample_summary}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Basin & Storms & Hurricanes & Major & Landfalls & $\\leq$100 km & Mean $\\phi_{\\mathrm{LMI}}$ & Median $d_{\\min}$ \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Basin']} & {fmt(r['Storms'],0)} & {fmt(r['Hurricanes'],0)} & "
            f"{fmt(r['Major hurricanes'],0)} & {fmt(r['Landfalls'],0)} & {fmt(r['Within 100 km'],0)} & "
            f"{fmt(r['Mean LMI lat'],1)} & {fmt(r['Median dmin'],1)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_trend_summary(trends: pd.DataFrame) -> str:
    keep_metrics = ["n_100km", "frac_100km", "phi_lmi_mean", "dmin_median_km", "ace_6h_sum", "translation_speed_min_mean"]
    names = {
        "n_100km": "Count within 100 km",
        "frac_100km": "Fraction within 100 km",
        "phi_lmi_mean": "Mean $\\phi_{\\mathrm{LMI}}$",
        "dmin_median_km": "Median $d_{\\min}$",
        "ace_6h_sum": "Annual ACE",
        "translation_speed_min_mean": "Mean coastal translation speed",
    }
    sub = trends[trends["metric"].isin(keep_metrics)].copy()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Trend diagnostics with Sen slope, residual moving-block bootstrap and Hamed--Rao autocorrelation-corrected Mann--Kendall diagnostics computed from Sen-detrended ranks. Slopes are per decade; HR probabilities are nominal.}",
        "\\label{tab:trend_summary}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Basin & Metric & Sen & 95\\% low & 95\\% high & HR $p$ & lag-1 $r$ \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["basin", "metric"]).iterrows():
        lines.append(
            f"{LABELS[r['basin']]} & {names[r['metric']]} & "
            f"{fmt(r['sen_slope_per_decade'],3)} & {fmt(r['boot_ci_low_decade'],3)} & "
            f"{fmt(r['boot_ci_high_decade'],3)} & {fmt(r['hr_p'],3)} & {fmt(r['lag1_acf'],2)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_glm_summary(glm: pd.DataFrame) -> str:
    sub = glm[glm["threshold_km"].isin([50, 100, 200, 300])].copy()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Primary quasi-binomial GLM tests for annual approach probability. Coefficients are logit trends per decade; dispersion inflates the binomial standard error when needed, is lower-bounded at 1.00, and Holm/BH-FDR adjustments are applied across the eight primary basin-threshold tests.}",
        "\\label{tab:glm_primary}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Basin & Threshold & Trend & qSE & Disp. & quasi $p$ & Holm $p$ & BH-FDR $p$ \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["basin", "threshold_km"]).iterrows():
        lines.append(
            f"{LABELS[r['basin']]} & {fmt(r['threshold_km'],0)} km & {fmt(r['coef_year_decade'],3)} & "
            f"{fmt(r['se_year_quasi'],3)} & {fmt(r['dispersion'],2)} & {fmt(r['p_year'],3)} & "
            f"{fmt(r['p_holm'],3)} & {fmt(r['p_fdr_bh'],3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_wasserstein(wass: pd.DataFrame) -> str:
    sub = wass[
        wass["metric"].isin(
            ["dmin_coast_km", "phi_lmi", "phi_gen", "vmax_lifetime", "ace_6h", "ace_per_day", "translation_speed_min_kmh"]
        )
    ].copy()
    names = {
        "dmin_coast_km": "$d_{\\min}$",
        "phi_lmi": "$\\phi_{\\mathrm{LMI}}$",
        "phi_gen": "$\\phi_{\\mathrm{gen}}$",
        "vmax_lifetime": "$V_{\\max}$",
        "ace_6h": "Storm ACE",
        "ace_per_day": "ACE per day",
        "translation_speed_min_kmh": "Coastal translation speed",
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Early--recent empirical distributional distances, 1949--1987 vs. 1988--2025, with bootstrap uncertainty for $W_1$. Intervals describe sampling variability and are not null-test intervals for equality.}",
        "\\label{tab:wasserstein}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Basin & Metric & $W_1$ & 95\\% low & 95\\% high & Early mean & Recent mean \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["basin", "metric"]).iterrows():
        lines.append(
            f"{LABELS[r['basin']]} & {names[r['metric']]} & {fmt(r['wasserstein'],2)} & "
            f"{fmt(r['w1_ci_low'],2)} & {fmt(r['w1_ci_high'],2)} & "
            f"{fmt(r['early_mean'],2)} & {fmt(r['recent_mean'],2)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_sensitivity(sens: pd.DataFrame) -> str:
    sub = sens[sens["threshold_km"].isin([100, 200])].copy()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Sensitivity of approach-probability trends to intensity subset. Coefficients are quasi-binomial logit trends per decade; BH-FDR-all is adjusted across the full table.}",
        "\\label{tab:intensity_sensitivity}",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Subset & Basin & Threshold & Storms & Events & Trend & quasi $p$ & BH-FDR-all \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["intensity_subset", "basin", "threshold_km"]).iterrows():
        lines.append(
            f"{r['intensity_subset']} & {LABELS[r['basin']]} & {fmt(r['threshold_km'],0)} km & {fmt(r['total_storms'],0)} & "
            f"{fmt(r['total_events'],0)} & {fmt(r['coef_year_decade'],3)} & {fmt(r['p_year'],3)} & {fmt(r['p_fdr_bh_all'],3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_era_sensitivity(era: pd.DataFrame) -> str:
    sub = era[era["threshold_km"].isin([100, 200])].copy()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Sensitivity of primary approach-probability trends to observational start year. Coefficients are quasi-binomial logit trends per decade. BH-FDR-by-start is adjusted within each start-year family; BH-FDR-all and Holm-all are adjusted across all 16 start-year sensitivity tests.}",
        "\\label{tab:era_sensitivity}",
        "\\begin{tabular}{rllrrrrrr}",
        "\\toprule",
        "Start & Basin & Threshold & Storms & Events & Trend & $p_{\\rm BH,start}$ & $p_{\\rm BH,all}$ & $p_{\\rm Holm,all}$ \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["start_year", "basin", "threshold_km"]).iterrows():
        lines.append(
            f"{int(r['start_year'])} & {LABELS[r['basin']]} & {fmt(r['threshold_km'],0)} km & "
            f"{fmt(r['total_storms'],0)} & {fmt(r['total_events'],0)} & {fmt(r['coef_year_decade'],3)} & "
            f"{fmt(r['p_fdr_bh'],3)} & {fmt(r['p_fdr_bh_all'],3)} & {fmt(r['p_holm_all'],3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_pettitt_summary(pettitt: pd.DataFrame) -> str:
    names = {
        "frac_100km": "Fraction within 100 km",
        "frac_200km": "Fraction within 200 km",
        "phi_lmi_mean": "Mean $\\phi_{\\mathrm{LMI}}$",
        "dmin_median_km": "Median $d_{\\min}$",
    }
    sub = pettitt[pettitt["metric"].isin(names)].copy()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\caption{Exploratory Pettitt change-point diagnostics for selected annual series. The year is the estimated most likely step-change location; $p$ is an approximate two-sided probability, unadjusted for the 12 series searched.}",
        "\\label{tab:pettitt_summary}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Basin & Metric & Pettitt year & $p$ \\\\",
        "\\midrule",
    ]
    for _, r in sub.sort_values(["basin", "metric"]).iterrows():
        lines.append(f"{LABELS[r['basin']]} & {names[r['metric']]} & {int(float(r['pettitt_year']))} & {fmt_p(r['pettitt_p'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_block_sensitivity(blocks: pd.DataFrame) -> str:
    names = {"frac_100km": "Fraction within 100 km", "frac_200km": "Fraction within 200 km"}
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Sensitivity of Sen-slope uncertainty to moving-block bootstrap length. Confidence intervals are based on the sensitivity bootstrap run.}",
        "\\label{tab:block_sensitivity}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Basin & Metric & Block & Sen & 95\\% low & 95\\% high \\\\",
        "\\midrule",
    ]
    for _, r in blocks.sort_values(["basin", "metric", "block_years"]).iterrows():
        lines.append(
            f"{LABELS[r['basin']]} & {names[r['metric']]} & {fmt(r['block_years'],0)} yr & "
            f"{fmt(r['sen_slope_per_decade'],3)} & {fmt(r['boot_ci_low_decade'],3)} & "
            f"{fmt(r['boot_ci_high_decade'],3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_reproducibility(storms: pd.DataFrame, annual: pd.DataFrame) -> str:
    obs_path = PROCESSED / "ibtracs_observation_panel_1949_2025.parquet"
    obs = pd.read_parquet(obs_path) if obs_path.exists() else pd.DataFrame()
    rows = [
        ("Source archive", "IBTrACS version 4r01 NetCDF"),
        ("Local source file", "IBTrACS\\_updated\\_20260209.nc"),
        ("Analysis period", f"{int(annual['season'].min())}--{int(annual['season'].max())}"),
        ("Basins", "North Atlantic; Eastern North Pacific"),
        ("Storm filter", "$V_{\\max}\\geq 34$ kt during lifetime"),
        ("Storms retained", fmt(len(storms), 0)),
        ("Active observations retained", fmt(len(obs), 0)),
        ("Basin-year rows", fmt(len(annual), 0)),
        ("Distance variables", "\\texttt{dist2land}; \\texttt{landfall}"),
        ("Primary thresholds", "50, 100, 200, 300 km"),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\caption{Reproducibility summary for the IBTrACS-only analysis.}",
        "\\label{tab:reproducibility}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Item & Value \\\\",
        "\\midrule",
    ]
    lines.extend(f"{item} & {value} \\\\" for item, value in rows)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_exposure_metrics(storms: pd.DataFrame, annual: pd.DataFrame) -> str:
    rows = []
    for basin, sub in storms.groupby("basin"):
        ann = annual[annual["basin"].eq(basin)]
        near100 = sub[sub["approach_100km"]]
        rows.append(
            {
                "Basin": LABELS[basin],
                "Total ACE": sub["ace_6h"].sum(),
                "Annual ACE": ann["ace_6h_sum"].mean(),
                "Coastal ACE 200": sub["coastal_ace_200km"].sum(),
                "Mean speed": near100["translation_speed_min_kmh"].mean(),
                "RI24 near coast": int(near100["ri_24h_before_min_approach"].sum()),
                "Cat3+ at min": int((near100["vmax_at_min_approach"] >= 96).sum()),
            }
        )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Additional IBTrACS-only coastal-hazard descriptors by basin, 1949--2025. Coastal speed, rapid intensification and category counts are computed for storms approaching within 100 km.}",
        "\\label{tab:exposure_metrics}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Basin & Total ACE & Mean annual ACE & Coastal ACE & Mean speed & RI24 & Cat3+ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Basin']} & {fmt(r['Total ACE'],1)} & {fmt(r['Annual ACE'],1)} & {fmt(r['Coastal ACE 200'],1)} & "
            f"{fmt(r['Mean speed'],1)} & {fmt(r['RI24 near coast'],0)} & {fmt(r['Cat3+ at min'],0)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_regional_summary(storms: pd.DataFrame) -> str:
    sub = storms[storms["dmin_coast_km"].le(100)].copy()
    named_regions = {
        "EP": ["Mexico Pacific", "Central America Pacific"],
        "NA": ["United States Gulf-Atlantic", "Mexico Gulf-Caribbean", "Caribbean and Central America"],
    }
    sub["display_region"] = sub.apply(
        lambda r: r["coastal_region"] if r["coastal_region"] in named_regions.get(r["basin"], []) else "Other/unclassified",
        axis=1,
    )
    reg = (
        sub.groupby(["basin", "display_region"])
        .agg(
            storms=("storm_id", "nunique"),
            landfalls=("landfall_ibtracs", "sum"),
            cat3plus=("vmax_at_min_approach", lambda x: int((x >= 96).sum())),
            ri24=("ri_24h_before_min_approach", "sum"),
        )
        .reset_index()
    )
    rows = []
    order = {
        "EP": ["Mexico Pacific", "Central America Pacific", "Other/unclassified"],
        "NA": ["United States Gulf-Atlantic", "Mexico Gulf-Caribbean", "Caribbean and Central America", "Other/unclassified"],
    }
    for basin in ["EP", "NA"]:
        total = reg[reg["basin"].eq(basin)]["storms"].sum()
        for region in order[basin]:
            match = reg[(reg["basin"].eq(basin)) & (reg["display_region"].eq(region))]
            if match.empty:
                vals = {"storms": 0, "landfalls": 0, "cat3plus": 0, "ri24": 0}
            else:
                vals = match.iloc[0].to_dict()
            rows.append(
                {
                    "basin": LABELS[basin],
                    "region": region,
                    "storms": int(vals["storms"]),
                    "share": 100.0 * int(vals["storms"]) / total if total else np.nan,
                    "landfalls": int(vals["landfalls"]),
                    "cat3plus": int(vals["cat3plus"]),
                    "ri24": int(vals["ri24"]),
                }
            )
        totals = reg[reg["basin"].eq(basin)].sum(numeric_only=True)
        rows.append(
            {
                "basin": LABELS[basin],
                "region": "\\textbf{Basin total}",
                "storms": int(totals["storms"]),
                "share": 100.0,
                "landfalls": int(totals["landfalls"]),
                "cat3plus": int(totals["cat3plus"]),
                "ri24": int(totals["ri24"]),
            }
        )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Storms approaching within 100 km by approximate coastal region. Each storm is assigned once, from its IBTrACS minimum-approach point, so basin totals reconcile with Table~\\ref{tab:sample_summary}. The Other/unclassified row contains coastal approaches outside the named planning regions.}",
        "\\label{tab:regional_summary}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Basin & Region & $\\leq$100 km & \\% basin & Landfalls & Cat3+ & RI24 \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['basin']} & {r['region']} & {fmt(r['storms'],0)} & {fmt(r['share'],1)} & {fmt(r['landfalls'],0)} & "
            f"{fmt(r['cat3plus'],0)} & {fmt(r['ri24'],0)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    storms = pd.read_parquet(PROCESSED / "ibtracs_storm_panel_1949_2025.parquet")
    annual = pd.read_parquet(PROCESSED / "annual_coastal_exposure_1949_2025.parquet")
    sample = sample_summary(storms)
    sample.to_csv(TABLE_DIR / "sample_summary.csv", index=False)
    write_text(TABLE_DIR / "sample_summary.tex", latex_sample_summary(sample))
    write_text(TABLE_DIR / "reproducibility_summary.tex", latex_reproducibility(storms, annual))
    write_text(TABLE_DIR / "exposure_metrics.tex", latex_exposure_metrics(storms, annual))
    write_text(TABLE_DIR / "regional_summary.tex", latex_regional_summary(storms))

    trend_path = OUTPUTS / "trend_bootstrap_results.csv"
    if trend_path.exists():
        trends = pd.read_csv(trend_path, keep_default_na=False)
        write_text(TABLE_DIR / "trend_summary.tex", latex_trend_summary(trends))

    glm_path = OUTPUTS / "binomial_glm_results.csv"
    if glm_path.exists():
        glm = pd.read_csv(glm_path, keep_default_na=False)
        write_text(TABLE_DIR / "glm_primary.tex", latex_glm_summary(glm))

    wass_path = OUTPUTS / "wasserstein_results.csv"
    if wass_path.exists():
        wass = pd.read_csv(wass_path, keep_default_na=False)
        write_text(TABLE_DIR / "wasserstein_summary.tex", latex_wasserstein(wass))

    sens_path = OUTPUTS / "intensity_sensitivity_glm.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path, keep_default_na=False)
        write_text(TABLE_DIR / "intensity_sensitivity.tex", latex_sensitivity(sens))

    era_path = OUTPUTS / "era_sensitivity_glm.csv"
    if era_path.exists():
        era = pd.read_csv(era_path, keep_default_na=False)
        write_text(TABLE_DIR / "era_sensitivity.tex", latex_era_sensitivity(era))

    block_path = OUTPUTS / "bootstrap_block_sensitivity.csv"
    if block_path.exists():
        blocks = pd.read_csv(block_path, keep_default_na=False)
        write_text(TABLE_DIR / "block_sensitivity.tex", latex_block_sensitivity(blocks))

    pettitt_path = OUTPUTS / "pettitt_change_point.csv"
    if pettitt_path.exists():
        pettitt = pd.read_csv(pettitt_path, keep_default_na=False)
        write_text(TABLE_DIR / "pettitt_summary.tex", latex_pettitt_summary(pettitt))

    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
