"""Run the coastal-proximity statistical analysis.

Methods implemented:
- Mann-Kendall test and Sen slope;
- moving-block bootstrap confidence intervals for Sen slopes;
- Wasserstein distances between early and recent periods;
- Monte Carlo power for detecting trends in coastal-approach probabilities;
- binomial GLM with annual storm-count weights for approach fractions.
- multiple-testing control for families of GLM trend tests;
- observational-era and block-length sensitivity checks.
- Hamed-Rao modified Mann-Kendall diagnostics for autocorrelated annual series;
- quasi-binomial variance inflation for annual approach-probability GLMs;
- bootstrap uncertainty and split sensitivity for Wasserstein distances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


ARTICLE = Path(__file__).resolve().parents[1]
PROCESSED = ARTICLE / "data" / "processed"
OUTPUTS = ARTICLE / "outputs"


def stable_seed(*parts: object, base: int = 20260705) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return base + int(digest[:8], 16) % 1_000_000


def mann_kendall_sen(year: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(year) & np.isfinite(y)
    x = np.asarray(year[mask], dtype=float)
    z = np.asarray(y[mask], dtype=float)
    n = len(z)
    if n < 8:
        return {
            "n": n,
            "sen_slope_per_year": np.nan,
            "sen_slope_per_decade": np.nan,
            "mk_z": np.nan,
            "mk_p": np.nan,
            "hr_z": np.nan,
            "hr_p": np.nan,
            "hr_var_factor": np.nan,
            "lag1_acf": np.nan,
        }

    s = 0
    slopes = []
    for i in range(n - 1):
        dz = z[i + 1 :] - z[i]
        dx = x[i + 1 :] - x[i]
        s += int(np.sign(dz).sum())
        slopes.extend((dz / dx).tolist())

    unique, counts = np.unique(z, return_counts=True)
    tie_term = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if s > 0:
        zmk = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        zmk = (s + 1) / np.sqrt(var_s)
    else:
        zmk = 0.0

    slope = float(np.nanmedian(slopes))
    # Hamed--Rao estimates serial dependence from detrended ranks; otherwise a
    # genuine monotone slope itself appears as long-lag autocorrelation and can
    # be counted twice in the variance correction.
    detrended = z - slope * x
    ranks = pd.Series(detrended).rank(method="average").to_numpy(dtype=float)
    lag_acf = []
    for lag in range(1, n):
        a = ranks[:-lag]
        b = ranks[lag:]
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(rho):
            lag_acf.append((lag, rho))
    sig = 1.96 / np.sqrt(n)
    var_factor = 1.0
    if lag_acf:
        correction = 0.0
        for lag, rho in lag_acf:
            if abs(rho) > sig:
                correction += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho
        var_factor = 1.0 + 2.0 * correction / (n * (n - 1) * (n - 2))
        var_factor = float(max(var_factor, 0.25))
    var_s_hr = var_s * var_factor
    if s > 0:
        zhr = (s - 1) / np.sqrt(var_s_hr)
    elif s < 0:
        zhr = (s + 1) / np.sqrt(var_s_hr)
    else:
        zhr = 0.0

    return {
        "n": int(n),
        "sen_slope_per_year": slope,
        "sen_slope_per_decade": slope * 10.0,
        "mk_z": float(zmk),
        "mk_p": float(2.0 * (1.0 - norm.cdf(abs(zmk)))),
        "hr_z": float(zhr),
        "hr_p": float(2.0 * (1.0 - norm.cdf(abs(zhr)))),
        "hr_var_factor": float(var_factor),
        "lag1_acf": float(lag_acf[0][1]) if lag_acf else np.nan,
    }


def moving_block_bootstrap_sen(year: np.ndarray, y: np.ndarray, block: int = 5, reps: int = 2000, seed: int = 20260705) -> dict[str, float]:
    """Residual moving-block bootstrap interval for Sen's slope.

    Resampling the raw series would break the fitted temporal trend and produce
    a null randomization distribution rather than a bootstrap distribution of
    the estimator.  We therefore remove the Sen trend, resample contiguous
    residual blocks, restore the fitted trend on the original years, and refit.
    """
    mask = np.isfinite(year) & np.isfinite(y)
    x = np.asarray(year[mask], dtype=float)
    z = np.asarray(y[mask], dtype=float)
    n = len(z)
    if n < block * 2:
        return {"boot_ci_low_decade": np.nan, "boot_ci_high_decade": np.nan, "boot_reps": 0}

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block + 1)
    ii, jj = np.triu_indices(n, k=1)
    dx = x[jj] - x[ii]
    fitted_slope = float(np.nanmedian((z[jj] - z[ii]) / dx))
    fitted_intercept = float(np.nanmedian(z - fitted_slope * x))
    residual = z - (fitted_intercept + fitted_slope * x)
    residual -= np.nanmean(residual)
    slopes = np.empty(reps, dtype=float)
    for _ in range(reps):
        draw = []
        while len(draw) < n:
            s = int(rng.choice(starts))
            draw.extend(range(s, s + block))
        idx = np.asarray(draw[:n])
        sampled = fitted_intercept + fitted_slope * x + residual[idx]
        pair_slopes = (sampled[jj] - sampled[ii]) / dx
        slopes[_] = np.nanmedian(pair_slopes) * 10.0
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return {"boot_ci_low_decade": float(lo), "boot_ci_high_decade": float(hi), "boot_reps": int(reps)}


def weighted_binomial_glm(annual: pd.DataFrame, event_col: str, total_col: str = "n_storms") -> dict[str, float]:
    df = annual[[event_col, total_col, "season"]].copy()
    df = df[df[total_col] > 0].dropna()
    if len(df) < 20:
        return {
            "coef_year_decade": np.nan,
            "se_year": np.nan,
            "se_year_quasi": np.nan,
            "dispersion": np.nan,
            "p_binomial": np.nan,
            "p_year": np.nan,
            "n": int(len(df)),
        }
    y = df[event_col] / df[total_col]
    x = (df["season"] - df["season"].mean()) / 10.0
    X = sm.add_constant(x)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=df[total_col])
        fit = model.fit()
        df_resid = max(float(fit.df_resid), 1.0)
        dispersion = float(max(fit.pearson_chi2 / df_resid, 1.0))
        se_quasi = float(fit.bse.iloc[1] * np.sqrt(dispersion))
        z_quasi = float(fit.params.iloc[1] / se_quasi) if se_quasi > 0 else np.nan
        p_quasi = float(2.0 * (1.0 - norm.cdf(abs(z_quasi)))) if np.isfinite(z_quasi) else np.nan
        return {
            "coef_year_decade": float(fit.params.iloc[1]),
            "se_year": float(fit.bse.iloc[1]),
            "se_year_quasi": se_quasi,
            "dispersion": dispersion,
            "p_binomial": float(fit.pvalues.iloc[1]),
            "p_year": p_quasi,
            "n": int(len(df)),
        }
    except Exception:
        return {
            "coef_year_decade": np.nan,
            "se_year": np.nan,
            "se_year_quasi": np.nan,
            "dispersion": np.nan,
            "p_binomial": np.nan,
            "p_year": np.nan,
            "n": int(len(df)),
        }


def bootstrap_wasserstein(a: np.ndarray, b: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"w1_ci_low": np.nan, "w1_ci_high": np.nan, "w1_boot_reps": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals[i] = wasserstein_distance(aa, bb)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"w1_ci_low": float(lo), "w1_ci_high": float(hi), "w1_boot_reps": int(reps)}


def pettitt_test(year: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(year) & np.isfinite(y)
    x = np.asarray(year[mask], dtype=float)
    z = np.asarray(y[mask], dtype=float)
    n = len(z)
    if n < 10:
        return {"pettitt_year": np.nan, "pettitt_k": np.nan, "pettitt_p": np.nan}
    ranks = pd.Series(z).rank(method="average").to_numpy(dtype=float)
    u = np.array([2.0 * np.sum(ranks[: k + 1]) - (k + 1) * (n + 1) for k in range(n)], dtype=float)
    pos = int(np.nanargmax(np.abs(u)))
    k_stat = float(abs(u[pos]))
    p = float(min(1.0, 2.0 * np.exp((-6.0 * k_stat**2) / (n**3 + n**2))))
    return {"pettitt_year": int(x[pos]), "pettitt_k": k_stat, "pettitt_p": p}


def add_multiplicity_adjustments(df: pd.DataFrame, p_col: str = "p_year", family_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    out["p_holm"] = np.nan
    out["p_fdr_bh"] = np.nan
    out["reject_holm_05"] = False
    out["reject_fdr_bh_05"] = False
    if family_cols is None:
        groups = [((), out.index)]
    else:
        grouper = family_cols[0] if len(family_cols) == 1 else family_cols
        groups = out.groupby(grouper, dropna=False).groups.items()
    for _, idx in groups:
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        valid = p.notna()
        if not valid.any():
            continue
        valid_idx = p.index[valid]
        pvals = p.loc[valid_idx].to_numpy(dtype=float)
        reject_holm, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        reject_fdr, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        out.loc[valid_idx, "p_holm"] = p_holm
        out.loc[valid_idx, "p_fdr_bh"] = p_fdr
        out.loc[valid_idx, "reject_holm_05"] = reject_holm
        out.loc[valid_idx, "reject_fdr_bh_05"] = reject_fdr
    return out


def monte_carlo_power(
    totals: np.ndarray,
    p0: float,
    delta_grid: np.ndarray,
    reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260705,
) -> pd.DataFrame:
    totals = np.asarray(totals, dtype=int)
    x = np.arange(len(totals), dtype=float)
    x = (x - x.mean()) / 10.0
    rng = np.random.default_rng(seed)
    records = []
    for delta in delta_grid:
        if abs(delta) < 1e-12:
            continue
        p1 = float(np.clip(p0 + delta, 0.01, 0.99))
        p = np.linspace(p0, p1, len(totals))

        # Vectorized score test for the slope in a binomial-logit model with an
        # intercept. This is the same null tested by the GLM trend term and is
        # far faster than fitting one GLM per Monte Carlo replicate.
        y = rng.binomial(totals[None, :], p[None, :], size=(reps, len(totals)))
        n_tot = totals.sum()
        p_hat = np.clip(y.sum(axis=1) / n_tot, 1e-6, 1.0 - 1e-6)
        xbar = np.sum(totals * x) / n_tot
        xc = x - xbar
        score = np.sum((y - p_hat[:, None] * totals[None, :]) * xc[None, :], axis=1)
        info = p_hat * (1.0 - p_hat) * np.sum(totals * xc**2)
        z = score / np.sqrt(info)
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))
        records.append({"delta_probability": float(delta), "power": float(np.mean(pvals < alpha)), "reps": reps})
    return pd.DataFrame(records)


def run_analysis(
    storms: pd.DataFrame,
    annual: pd.DataFrame,
    bootstrap_reps: int,
    mc_reps: int,
    sensitivity_reps: int,
) -> dict[str, object]:
    trend_metrics = [
        "n_storms",
        "n_landfall",
        "n_100km",
        "n_200km",
        "frac_100km",
        "frac_200km",
        "phi_lmi_mean",
        "phi_lmi_median",
        "dmin_median_km",
    ]
    trend_rows = []
    for basin, sub in annual.groupby("basin"):
        for metric in trend_metrics:
            mk = mann_kendall_sen(sub["season"].to_numpy(), sub[metric].to_numpy(dtype=float))
            boot = moving_block_bootstrap_sen(
                sub["season"].to_numpy(),
                sub[metric].to_numpy(dtype=float),
                reps=bootstrap_reps,
                seed=stable_seed("trend", basin, metric),
            )
            trend_rows.append({"basin": basin, "metric": metric, **mk, **boot})

    glm_rows = []
    for basin, sub in annual.groupby("basin"):
        for threshold in [50, 100, 200, 300]:
            event_col = f"n_{threshold}km"
            glm_rows.append({"basin": basin, "threshold_km": threshold, **weighted_binomial_glm(sub, event_col)})

    sensitivity_rows = []
    for min_wind, label in [(34.0, "TS+"), (64.0, "HU+"), (96.0, "Major+")]:
        sub_storms = storms[storms["vmax_lifetime"] >= min_wind].copy()
        years = pd.MultiIndex.from_product(
            [["NA", "EP"], range(int(annual["season"].min()), int(annual["season"].max()) + 1)],
            names=["basin", "season"],
        ).to_frame(index=False)
        grouped = sub_storms.groupby(["basin", "season"], dropna=False)
        ann_i = grouped.agg(
            n_storms=("storm_id", "count"),
            n_50km=("approach_50km", "sum"),
            n_100km=("approach_100km", "sum"),
            n_200km=("approach_200km", "sum"),
            n_300km=("approach_300km", "sum"),
        ).reset_index()
        ann_i = years.merge(ann_i, on=["basin", "season"], how="left")
        for col in ["n_storms", "n_50km", "n_100km", "n_200km", "n_300km"]:
            ann_i[col] = ann_i[col].fillna(0).astype(int)
        for basin, sub in ann_i.groupby("basin"):
            total_storms = int(sub["n_storms"].sum())
            for threshold in [100, 200]:
                event_col = f"n_{threshold}km"
                stat = weighted_binomial_glm(sub, event_col)
                sensitivity_rows.append(
                    {
                        "intensity_subset": label,
                        "min_wind_kt": min_wind,
                        "basin": basin,
                        "threshold_km": threshold,
                        "total_storms": total_storms,
                        "total_events": int(sub[event_col].sum()),
                        **stat,
                    }
                )

    era_rows = []
    for start_year in [1949, 1966, 1970, 1980]:
        era = annual[annual["season"] >= start_year].copy()
        for basin, sub in era.groupby("basin"):
            total_storms = int(sub["n_storms"].sum())
            for threshold in [100, 200]:
                event_col = f"n_{threshold}km"
                stat = weighted_binomial_glm(sub, event_col)
                era_rows.append(
                    {
                        "start_year": start_year,
                        "end_year": int(sub["season"].max()),
                        "basin": basin,
                        "threshold_km": threshold,
                        "total_storms": total_storms,
                        "total_events": int(sub[event_col].sum()),
                        **stat,
                    }
                )

    block_rows = []
    primary_metrics = ["frac_100km", "frac_200km"]
    for basin, sub in annual.groupby("basin"):
        for metric in primary_metrics:
            mk = mann_kendall_sen(sub["season"].to_numpy(), sub[metric].to_numpy(dtype=float))
            for block in [3, 5, 7, 10]:
                boot = moving_block_bootstrap_sen(
                    sub["season"].to_numpy(),
                    sub[metric].to_numpy(dtype=float),
                    block=block,
                    reps=sensitivity_reps,
                    seed=stable_seed("block", basin, metric, block),
                )
                block_rows.append(
                    {
                        "basin": basin,
                        "metric": metric,
                        "block_years": block,
                        "sen_slope_per_decade": mk["sen_slope_per_decade"],
                        **boot,
                    }
                )

    wass_rows = []
    split_rows = []
    period_mid = 1987
    for basin, sub in storms.groupby("basin"):
        early = sub[sub["season"] <= period_mid]
        recent = sub[sub["season"] > period_mid]
        for metric in [
            "dmin_coast_km",
            "phi_lmi",
            "phi_gen",
            "vmax_lifetime",
            "ace_6h",
            "ace_per_day",
            "translation_speed_min_kmh",
        ]:
            a = early[metric].dropna().to_numpy(dtype=float)
            b = recent[metric].dropna().to_numpy(dtype=float)
            dist = float(wasserstein_distance(a, b)) if len(a) and len(b) else np.nan
            boot_w1 = bootstrap_wasserstein(
                a,
                b,
                reps=sensitivity_reps,
                seed=stable_seed("w1", basin, metric, period_mid),
            )
            wass_rows.append(
                {
                    "basin": basin,
                    "metric": metric,
                    "early_period": "1949-1987",
                    "recent_period": "1988-2025",
                    "n_early": int(len(a)),
                    "n_recent": int(len(b)),
                    "wasserstein": dist,
                    **boot_w1,
                    "early_mean": float(np.nanmean(a)) if len(a) else np.nan,
                    "recent_mean": float(np.nanmean(b)) if len(b) else np.nan,
                }
            )
            for split_year in range(1965, 2006, 5):
                early_s = sub[sub["season"] <= split_year][metric].dropna().to_numpy(dtype=float)
                recent_s = sub[sub["season"] > split_year][metric].dropna().to_numpy(dtype=float)
                if len(early_s) and len(recent_s):
                    split_rows.append(
                        {
                            "basin": basin,
                            "metric": metric,
                            "split_year": split_year,
                            "n_early": int(len(early_s)),
                            "n_recent": int(len(recent_s)),
                            "wasserstein": float(wasserstein_distance(early_s, recent_s)),
                        }
                    )

    pettitt_rows = []
    for basin, sub in annual.groupby("basin"):
        for metric in ["frac_100km", "frac_200km", "phi_lmi_mean", "dmin_median_km", "ace_6h_sum", "translation_speed_min_mean"]:
            stat = pettitt_test(sub["season"].to_numpy(), sub[metric].to_numpy(dtype=float))
            pettitt_rows.append({"basin": basin, "metric": metric, **stat})

    power_rows = []
    delta_grid = np.round(np.r_[np.linspace(-0.30, -0.02, 15), np.linspace(0.02, 0.30, 15)], 3)
    for basin, sub in annual.groupby("basin"):
        totals = sub["n_storms"].to_numpy(dtype=int)
        valid_totals = totals[totals > 0]
        if len(valid_totals) < 20:
            continue
        for threshold in [100, 200]:
            frac = sub.loc[sub["n_storms"] > 0, f"frac_{threshold}km"].dropna()
            p0 = float(np.clip(frac.iloc[: min(20, len(frac))].mean(), 0.01, 0.99))
            power = monte_carlo_power(totals, p0, delta_grid, reps=mc_reps, seed=20260705 + threshold)
            power["basin"] = basin
            power["threshold_km"] = threshold
            power["baseline_probability"] = p0
            power_rows.append(power)

    trend_df = pd.DataFrame(trend_rows)
    glm_df = add_multiplicity_adjustments(pd.DataFrame(glm_rows))
    sensitivity_df = add_multiplicity_adjustments(pd.DataFrame(sensitivity_rows), family_cols=["intensity_subset"])
    if not sensitivity_df.empty:
        valid_p = sensitivity_df["p_year"].notna()
        sensitivity_df["p_fdr_bh_all"] = np.nan
        sensitivity_df["p_holm_all"] = np.nan
        if valid_p.any():
            pvals = sensitivity_df.loc[valid_p, "p_year"].to_numpy(dtype=float)
            _, p_holm_all, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            _, p_fdr_all, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            sensitivity_df.loc[valid_p, "p_holm_all"] = p_holm_all
            sensitivity_df.loc[valid_p, "p_fdr_bh_all"] = p_fdr_all
    era_df = add_multiplicity_adjustments(pd.DataFrame(era_rows), family_cols=["start_year"])
    if not era_df.empty:
        valid_p = era_df["p_year"].notna()
        era_df["p_fdr_bh_all"] = np.nan
        era_df["p_holm_all"] = np.nan
        era_df["reject_fdr_bh_all_05"] = False
        era_df["reject_holm_all_05"] = False
        if valid_p.any():
            pvals = era_df.loc[valid_p, "p_year"].to_numpy(dtype=float)
            reject_holm_all, p_holm_all, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            reject_fdr_all, p_fdr_all, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            era_df.loc[valid_p, "p_holm_all"] = p_holm_all
            era_df.loc[valid_p, "p_fdr_bh_all"] = p_fdr_all
            era_df.loc[valid_p, "reject_holm_all_05"] = reject_holm_all
            era_df.loc[valid_p, "reject_fdr_bh_all_05"] = reject_fdr_all
    block_df = pd.DataFrame(block_rows)
    wass_df = pd.DataFrame(wass_rows)
    split_df = pd.DataFrame(split_rows)
    pettitt_df = pd.DataFrame(pettitt_rows)
    power_df = pd.concat(power_rows, ignore_index=True) if power_rows else pd.DataFrame()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    trend_df.to_csv(OUTPUTS / "trend_bootstrap_results.csv", index=False)
    glm_df.to_csv(OUTPUTS / "binomial_glm_results.csv", index=False)
    sensitivity_df.to_csv(OUTPUTS / "intensity_sensitivity_glm.csv", index=False)
    era_df.to_csv(OUTPUTS / "era_sensitivity_glm.csv", index=False)
    block_df.to_csv(OUTPUTS / "bootstrap_block_sensitivity.csv", index=False)
    wass_df.to_csv(OUTPUTS / "wasserstein_results.csv", index=False)
    split_df.to_csv(OUTPUTS / "wasserstein_split_sensitivity.csv", index=False)
    pettitt_df.to_csv(OUTPUTS / "pettitt_change_point.csv", index=False)
    power_df.to_csv(OUTPUTS / "monte_carlo_power.csv", index=False)

    summary = {
        "bootstrap_reps": bootstrap_reps,
        "monte_carlo_reps": mc_reps,
        "sensitivity_bootstrap_reps": sensitivity_reps,
        "trend_results": "outputs/trend_bootstrap_results.csv",
        "binomial_glm_results": "outputs/binomial_glm_results.csv",
        "intensity_sensitivity_glm": "outputs/intensity_sensitivity_glm.csv",
        "era_sensitivity_glm": "outputs/era_sensitivity_glm.csv",
        "bootstrap_block_sensitivity": "outputs/bootstrap_block_sensitivity.csv",
        "wasserstein_results": "outputs/wasserstein_results.csv",
        "wasserstein_split_sensitivity": "outputs/wasserstein_split_sensitivity.csv",
        "pettitt_change_point": "outputs/pettitt_change_point.csv",
        "monte_carlo_power": "outputs/monte_carlo_power.csv",
    }
    (OUTPUTS / "statistical_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--mc-reps", type=int, default=20000)
    parser.add_argument("--sensitivity-reps", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storms = pd.read_parquet(PROCESSED / "ibtracs_storm_panel_1949_2025.parquet")
    annual = pd.read_parquet(PROCESSED / "annual_coastal_exposure_1949_2025.parquet")
    summary = run_analysis(storms, annual, args.bootstrap_reps, args.mc_reps, args.sensitivity_reps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
