"""Generate manuscript figures from processed IBTrACS panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

# Keep submitted vector artwork free of Type-3 glyphs.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

try:
    import geopandas as gpd
    from pyproj import Transformer
except ImportError:  # non-map figures remain reproducible without geospatial extras
    gpd = None
    Transformer = None


ARTICLE = Path(__file__).resolve().parents[1]
PROCESSED = ARTICLE / "data" / "processed"
OUTPUTS = ARTICLE / "outputs"
FIG_DIR = ARTICLE / "pdf" / "figures"
NATURAL_EARTH = ARTICLE / "data" / "external" / "natural_earth"
ALBERS_NORTH_AMERICA = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=30 +lon_0=-96 +datum=WGS84 +units=m +no_defs"


COLORS = {"NA": "#1f77b4", "EP": "#d95f02"}
LABELS = {"NA": "North Atlantic", "EP": "Eastern North Pacific"}
TRANSFORMER = (Transformer.from_crs("EPSG:4326", ALBERS_NORTH_AMERICA, always_xy=True)
               if Transformer is not None else None)


def savefig(path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300)
    # TCRR accepts PDF vector artwork and requires at least 500 dpi for any
    # rasterized line/halftone components embedded in that PDF.
    plt.savefig(path.with_suffix(".pdf"), dpi=600)
    plt.close()


def project_xy(lon: pd.Series | np.ndarray, lat: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if TRANSFORMER is None:
        raise RuntimeError("Map generation requires geopandas and pyproj")
    return TRANSFORMER.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def projected_extent(domain: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = domain
    # A conic projection bends the domain edges; using only four corners can
    # clip Mexico, Central America and the tropical Pacific.  Sample the full
    # perimeter so every requested longitude--latitude edge is represented.
    edge_lon = np.linspace(lon_min, lon_max, 181)
    edge_lat = np.linspace(lat_min, lat_max, 121)
    lons = np.concatenate([
        edge_lon, edge_lon,
        np.full_like(edge_lat, lon_min), np.full_like(edge_lat, lon_max),
    ])
    lats = np.concatenate([
        np.full_like(edge_lon, lat_min), np.full_like(edge_lon, lat_max),
        edge_lat, edge_lat,
    ])
    xs, ys = project_xy(lons, lats)
    pad_x = (np.nanmax(xs) - np.nanmin(xs)) * 0.03
    pad_y = (np.nanmax(ys) - np.nanmin(ys)) * 0.03
    return np.nanmin(xs) - pad_x, np.nanmax(xs) + pad_x, np.nanmin(ys) - pad_y, np.nanmax(ys) + pad_y


def set_map_axis(ax: plt.Axes, domain: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = projected_extent(domain)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_track_map(obs: pd.DataFrame, storms: pd.DataFrame) -> None:
    countries, coastline, ocean = load_basemap()
    countries = countries.to_crs(ALBERS_NORTH_AMERICA)
    coastline = coastline.to_crs(ALBERS_NORTH_AMERICA)
    ocean = ocean.to_crs(ALBERS_NORTH_AMERICA)
    fig, axes = plt.subplots(4, 2, figsize=(9.4, 10.8), sharex=True, sharey=True)
    sample = obs.sort_values(["storm_id", "iso_time"]).copy()
    sample["decade"] = sample["season"].floordiv(10).mul(10)
    decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    domain = (-170, -5, 0, 65)
    for ax, decade in zip(axes.ravel(), decades):
        ocean.plot(ax=ax, color="#c9e3f2", edgecolor="none", zorder=0,
                   rasterized=True)
        countries.plot(ax=ax, color="#f4f0e6", edgecolor="#c8c1b2",
                       linewidth=0.18, zorder=1, rasterized=True)
        coastline.plot(ax=ax, color="#5a5a5a", linewidth=0.22, zorder=2,
                       rasterized=True)
        sub_dec = sample[sample["decade"].eq(decade)]
        segments = []
        for _, trk in sub_dec.groupby("storm_id", sort=False):
            x, y = project_xy(trk["lon"], trk["lat"])
            segments.append(np.column_stack([x, y]))
        ax.add_collection(LineCollection(
            segments, colors="#34495e", linewidths=0.35, alpha=0.22,
            zorder=3, rasterized=True,
        ))
        lmi = storms[storms["season"].floordiv(10).mul(10).eq(decade)]
        if not lmi.empty:
            x_lmi, y_lmi = project_xy(lmi["lon_lmi"], lmi["phi_lmi"])
            ax.scatter(x_lmi, y_lmi, s=5, c=lmi["basin"].map(COLORS), alpha=0.55,
                       linewidths=0, zorder=4, rasterized=True)
        ax.set_title(f"{decade}s", fontsize=10)
        set_map_axis(ax, domain)
    fig.suptitle("IBTrACS intensity-filtered tracks and LMI locations by decade, 1950s-2020s", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.025, top=0.955, wspace=0.08, hspace=0.18)
    savefig(FIG_DIR / "fig01_tracks_lmi.png", tight=False)


def load_basemap() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    countries_path = NATURAL_EARTH / "countries_50m" / "ne_50m_admin_0_countries.shp"
    coastline_path = NATURAL_EARTH / "coastline_50m" / "ne_50m_coastline.shp"
    ocean_path = NATURAL_EARTH / "ocean_50m" / "ne_50m_ocean.shp"
    if gpd is None:
        raise RuntimeError("Map generation requires geopandas and pyproj")
    missing = [p for p in (countries_path, coastline_path, ocean_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Natural Earth basemap files are absent: " + ", ".join(map(str, missing)))
    return gpd.read_file(countries_path), gpd.read_file(coastline_path), gpd.read_file(ocean_path)


def plot_coastal_density_map(storms: pd.DataFrame) -> None:
    """Map storm-level coastal proximity using Natural Earth as basemap."""
    countries, coastline, ocean = load_basemap()
    countries = countries.to_crs(ALBERS_NORTH_AMERICA)
    coastline = coastline.to_crs(ALBERS_NORTH_AMERICA)
    ocean = ocean.to_crs(ALBERS_NORTH_AMERICA)
    domain = (-170, -5, 0, 65)
    coastal = storms[
        storms["dmin_coast_km"].le(200)
        & storms["min_approach_lon"].between(domain[0], domain[1])
        & storms["min_approach_lat"].between(domain[2], domain[3])
    ].copy()

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.set_facecolor("#dcecf7")
    ocean.plot(ax=ax, color="#c9e3f2", edgecolor="none", zorder=0)
    countries.plot(ax=ax, color="#f4f0e6", edgecolor="#b8b1a3", linewidth=0.35, zorder=1)
    x, y = project_xy(coastal["min_approach_lon"], coastal["min_approach_lat"])
    hb = ax.hexbin(
        x,
        y,
        gridsize=58,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.05,
        edgecolors="none",
        alpha=0.88,
        zorder=3,
        rasterized=True,
    )
    cbar = fig.colorbar(hb, ax=ax, shrink=0.78, pad=0.018)
    cbar.set_label("Distinct storms within 200 km (log count)")
    coastline.plot(ax=ax, color="#4a4a4a", linewidth=0.45, zorder=4)

    set_map_axis(ax, domain)
    ax.set_title("IBTrACS storm-level coastal-proximity density, NA and EP, 1949-2025")
    tx, ty = project_xy(np.array([-125, -72]), np.array([14, 34]))
    ax.text(tx[0], ty[0], "Eastern North Pacific", fontsize=9, weight="bold", color="#252525")
    ax.text(tx[1], ty[1], "North Atlantic", fontsize=9, weight="bold", color="#252525")
    savefig(FIG_DIR / "fig06_coastal_density_map.png")


def plot_regional_coastal_map(storms: pd.DataFrame) -> None:
    countries, coastline, ocean = load_basemap()
    countries = countries.to_crs(ALBERS_NORTH_AMERICA)
    coastline = coastline.to_crs(ALBERS_NORTH_AMERICA)
    ocean = ocean.to_crs(ALBERS_NORTH_AMERICA)
    domain = (-125, -55, 5, 46)
    sub = storms[
        storms["dmin_coast_km"].le(100)
        & storms["min_approach_lon"].between(domain[0], domain[1])
        & storms["min_approach_lat"].between(domain[2], domain[3])
    ].copy()
    region_order = [
        "Mexico Pacific",
        "Central America Pacific",
        "Mexico Gulf-Caribbean",
        "Caribbean and Central America",
        "United States Gulf-Atlantic",
    ]
    palette = {
        "Mexico Pacific": "#d95f02",
        "Central America Pacific": "#e6ab02",
        "Mexico Gulf-Caribbean": "#1b9e77",
        "Caribbean and Central America": "#7570b3",
        "United States Gulf-Atlantic": "#1f78b4",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_facecolor("#dcecf7")
    ocean.plot(ax=ax, color="#c9e3f2", edgecolor="none", zorder=0)
    countries.plot(ax=ax, color="#f4f0e6", edgecolor="#b8b1a3", linewidth=0.35, zorder=1)
    coastline.plot(ax=ax, color="#4a4a4a", linewidth=0.45, zorder=2)
    for region in region_order:
        r = sub[sub["coastal_region"].eq(region)]
        if r.empty:
            continue
        x, y = project_xy(r["min_approach_lon"], r["min_approach_lat"])
        ax.scatter(
            x,
            y,
            s=np.where(r["vmax_at_min_approach"] >= 96, 16, 8),
            color=palette[region],
            alpha=0.55,
            linewidths=0,
            label=f"{region} ({r['storm_id'].nunique()})",
            zorder=3,
            rasterized=True,
        )
    set_map_axis(ax, domain)
    ax.set_title("Storms approaching within 100 km by coastal planning region")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    savefig(FIG_DIR / "fig07_regional_coastal_approaches.png")


def plot_return_period_map(storms: pd.DataFrame) -> None:
    countries, coastline, ocean = load_basemap()
    countries = countries.to_crs(ALBERS_NORTH_AMERICA)
    coastline = coastline.to_crs(ALBERS_NORTH_AMERICA)
    ocean = ocean.to_crs(ALBERS_NORTH_AMERICA)
    domain = (-125, -55, 5, 46)
    sub = storms[
        storms["dmin_coast_km"].le(100)
        & storms["min_approach_lon"].between(domain[0], domain[1])
        & storms["min_approach_lat"].between(domain[2], domain[3])
    ].copy()
    x, y = project_xy(sub["min_approach_lon"], sub["min_approach_lat"])
    cell_m = 125_000.0
    sub["xbin"] = np.floor(x / cell_m).astype(int)
    sub["ybin"] = np.floor(y / cell_m).astype(int)
    sub["x"] = x
    sub["y"] = y
    grouped = (
        sub.groupby(["xbin", "ybin"])
        .agg(x=("x", "mean"), y=("y", "mean"), active_years=("season", "nunique"), storms=("storm_id", "count"))
        .reset_index()
    )
    n_years = int(storms["season"].max() - storms["season"].min() + 1)
    grouped["return_period"] = n_years / grouped["active_years"]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_facecolor("#dcecf7")
    ocean.plot(ax=ax, color="#c9e3f2", edgecolor="none", zorder=0)
    countries.plot(ax=ax, color="#f4f0e6", edgecolor="#b8b1a3", linewidth=0.35, zorder=1)
    coastline.plot(ax=ax, color="#4a4a4a", linewidth=0.45, zorder=2)
    sc = ax.scatter(
        grouped["x"],
        grouped["y"],
        c=grouped["return_period"],
        s=np.clip(grouped["storms"] * 8, 18, 120),
        cmap="viridis_r",
        vmin=1,
        vmax=20,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="#222222",
        zorder=3,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.018, extend="max")
    cbar.set_label("Approximate return period of <=100 km approach (years)")
    set_map_axis(ax, domain)
    ax.set_title("Approximate coastal approach return period, 1949-2025")
    savefig(FIG_DIR / "fig08_return_period_map.png")


def plot_early_recent_density(storms: pd.DataFrame) -> None:
    countries, coastline, ocean = load_basemap()
    countries = countries.to_crs(ALBERS_NORTH_AMERICA)
    coastline = coastline.to_crs(ALBERS_NORTH_AMERICA)
    ocean = ocean.to_crs(ALBERS_NORTH_AMERICA)
    domain = (-170, -5, 0, 65)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.4), sharex=True, sharey=True)
    periods = [("1949-1987", storms["season"].le(1987)), ("1988-2025", storms["season"].gt(1987))]
    for ax, (label, mask) in zip(axes, periods):
        sub = storms[mask & storms["dmin_coast_km"].le(200)].copy()
        x, y = project_xy(sub["min_approach_lon"], sub["min_approach_lat"])
        ax.set_facecolor("#dcecf7")
        ocean.plot(ax=ax, color="#c9e3f2", edgecolor="none", zorder=0)
        countries.plot(ax=ax, color="#f4f0e6", edgecolor="#c8c1b2", linewidth=0.25, zorder=1)
        hb = ax.hexbin(x, y, gridsize=46, mincnt=1, cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=30), linewidths=0,
                       alpha=0.88, zorder=2, rasterized=True)
        coastline.plot(ax=ax, color="#4a4a4a", linewidth=0.35, zorder=3)
        set_map_axis(ax, domain)
        ax.set_title(label)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.90, hspace=0.14)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label("Distinct storms within 200 km")
    fig.suptitle("Early and recent storm-level coastal-proximity density", y=0.96)
    savefig(FIG_DIR / "fig09_early_recent_density.png", tight=False)


def plot_annual_counts(annual: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    for basin, sub in annual.groupby("basin"):
        sub = sub.sort_values("season")
        axes[0].plot(sub["season"], sub["n_storms"], color=COLORS[basin], lw=1.2, label=LABELS[basin])
        axes[1].plot(sub["season"], sub["n_100km"], color=COLORS[basin], lw=1.2, label=LABELS[basin])
    axes[0].set_ylabel("Storms")
    axes[1].set_ylabel("Within 100 km")
    axes[1].set_xlabel("Season")
    axes[0].set_title("Annual basin counts and coastal approaches")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False)
    savefig(FIG_DIR / "fig02_annual_counts.png")


def plot_dmin_distribution(storms: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    grid = np.linspace(0, 800, 240)
    rng = np.random.default_rng(20260705)
    for ax, basin in zip(axes, ["NA", "EP"]):
        sub = storms[storms["basin"].eq(basin)]
        series = [
            ("1949-1987", sub[sub["season"] <= 1987]["dmin_coast_km"].dropna().to_numpy(dtype=float), "#7570b3"),
            ("1988-2025", sub[sub["season"] > 1987]["dmin_coast_km"].dropna().to_numpy(dtype=float), "#e7298a"),
        ]
        for label, vals, color in series:
            kde = gaussian_kde(vals)
            density = kde(grid)
            boot = []
            for _ in range(250):
                sample = rng.choice(vals, size=len(vals), replace=True)
                boot.append(gaussian_kde(sample)(grid))
            lo, hi = np.nanpercentile(np.vstack(boot), [2.5, 97.5], axis=0)
            ax.plot(grid, density, color=color, lw=1.6, label=label)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.set_title(LABELS[basin])
        ax.set_xlabel("Minimum distance to land (km)")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False)
    savefig(FIG_DIR / "fig03_dmin_distribution.png")


def plot_lmi_genesis(storms: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    for ax, basin in zip(axes, ["NA", "EP"]):
        sub = storms[storms["basin"].eq(basin)]
        ax.scatter(sub["phi_gen"], sub["phi_lmi"], s=10, alpha=0.45, color=COLORS[basin], linewidths=0)
        lim = [0, max(60, float(np.nanmax(sub[["phi_gen", "phi_lmi"]].to_numpy())) + 2)]
        ax.plot(lim, lim, color="0.4", ls="--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(LABELS[basin])
        ax.set_xlabel("Genesis latitude")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("LMI latitude")
    savefig(FIG_DIR / "fig04_genesis_lmi.png")


def plot_power() -> None:
    path = OUTPUTS / "monte_carlo_power.csv"
    if not path.exists():
        return
    power = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, threshold in zip(axes, [100, 200]):
        sub_t = power[power["threshold_km"].eq(threshold)]
        for basin, sub in sub_t.groupby("basin"):
            sub = sub.sort_values("delta_probability")
            style = "-" if basin == "EP" else "--"
            marker = "o" if basin == "EP" else "s"
            ax.plot(
                sub["delta_probability"],
                sub["power"],
                color=COLORS[basin],
                lw=1.5,
                ls=style,
                marker=marker,
                ms=2.2,
                markevery=4,
                label=LABELS[basin],
            )
        ax.axhline(0.8, color="0.3", ls="--", lw=1)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(f"Approach within {threshold} km")
        ax.set_xlabel("End-minus-start probability change")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("Monte Carlo power")
    axes[1].legend(frameon=False)
    savefig(FIG_DIR / "fig05_power.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-maps", action="store_true",
        help="Generate only non-map figures; do not replace existing map artwork.",
    )
    args = parser.parse_args()
    storms = pd.read_parquet(PROCESSED / "ibtracs_storm_panel_1949_2025.parquet")
    annual = pd.read_parquet(PROCESSED / "annual_coastal_exposure_1949_2025.parquet")
    obs_path = PROCESSED / "ibtracs_observation_panel_1949_2025.parquet"
    map_inputs = (
        gpd is not None
        and all((NATURAL_EARTH / rel).exists() for rel in [
            "countries_50m/ne_50m_admin_0_countries.shp",
            "coastline_50m/ne_50m_coastline.shp",
            "ocean_50m/ne_50m_ocean.shp",
        ])
    )
    if not args.skip_maps and (not obs_path.exists() or not map_inputs):
        parser.error(
            "Full figure regeneration requires the observation panel, geopandas, "
            "pyproj and Natural Earth files. See data/external/README.md. "
            "Use --skip-maps explicitly for a partial non-map rebuild."
        )
    if not args.skip_maps:
        obs = pd.read_parquet(obs_path)
        plot_track_map(obs, storms)
        plot_coastal_density_map(storms)
        plot_regional_coastal_map(storms)
        plot_return_period_map(storms)
        plot_early_recent_density(storms)
    else:
        print("Partial rebuild requested: map figures were not regenerated.")
    plot_annual_counts(annual)
    plot_dmin_distribution(storms)
    plot_lmi_genesis(storms)
    plot_power()
    scope = "Non-map figures" if args.skip_maps else "All manuscript figures"
    print(f"{scope} written to {FIG_DIR}")


if __name__ == "__main__":
    main()
