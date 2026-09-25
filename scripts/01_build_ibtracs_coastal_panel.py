"""Build the standalone IBTrACS-only coastal-proximity panels.

Outputs:
- data/processed/ibtracs_storm_panel_1949_2025.csv/parquet
- data/processed/ibtracs_observation_panel_1949_2025.parquet
- data/processed/annual_coastal_exposure_1949_2025.csv/parquet
- outputs/panel_summary.json

The coastal metric uses IBTrACS variables:
- dist2land: distance to land at the current observation;
- landfall: minimum distance to land between the current and next observation.
The storm-level coastal distance is min(dist2land, landfall) over the active
tropical-storm phase.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point


ARTICLE = Path(__file__).resolve().parents[1]
DEFAULT_IBTRACS = ARTICLE / "data" / "raw" / "ibtracs" / "IBTrACS_updated_20260209.nc"
NATURAL_EARTH_COUNTRIES = ARTICLE / "data" / "external" / "natural_earth" / "countries_50m" / "ne_50m_admin_0_countries.shp"
PROCESSED = ARTICLE / "data" / "processed"
OUTPUTS = ARTICLE / "outputs"
ALBERS_NORTH_AMERICA = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=30 +lon_0=-96 +datum=WGS84 +units=m +no_defs"

DEFAULT_START = 1949
DEFAULT_END = 2025
THRESHOLDS_KM = (50, 100, 200, 300)
INTENSITY_CLASSES = {
    "ts_plus": 34.0,
    "hu_plus": 64.0,
    "major_plus": 96.0,
}
EARTH_RADIUS_KM = 6371.0


def decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def decode_array(values: np.ndarray) -> np.ndarray:
    flat = [decode_scalar(v) for v in np.asarray(values).ravel()]
    return np.asarray(flat, dtype=object).reshape(np.asarray(values).shape)


def clean_numeric(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    arr = np.asarray(lon, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def decade_label(year: int) -> str:
    return f"{int(year) // 10 * 10}s"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def saffir_simpson_category(vmax_kt: float) -> str:
    if not np.isfinite(vmax_kt):
        return "unknown"
    if vmax_kt < 34:
        return "TD"
    if vmax_kt < 64:
        return "TS"
    if vmax_kt < 83:
        return "Cat1"
    if vmax_kt < 96:
        return "Cat2"
    if vmax_kt < 113:
        return "Cat3"
    if vmax_kt < 137:
        return "Cat4"
    return "Cat5"


def coastal_region(basin: str, lat_value: float, lon_value: float) -> str:
    if not np.isfinite(lat_value) or not np.isfinite(lon_value):
        return "Unclassified"
    if basin == "EP":
        if -118 <= lon_value <= -90 and 14 <= lat_value <= 33:
            return "Mexico Pacific"
        if -96 <= lon_value <= -77 and 5 <= lat_value < 18:
            return "Central America Pacific"
        return "Other EPac"
    if basin == "NA":
        if -100 <= lon_value <= -65 and 24 <= lat_value <= 46:
            return "United States Gulf-Atlantic"
        if -98 <= lon_value <= -81 and 18 <= lat_value < 24:
            return "Mexico Gulf-Caribbean"
        if -90 <= lon_value <= -60 and 8 <= lat_value <= 24:
            return "Caribbean and Central America"
        return "Other Atlantic"
    return "Unclassified"


def planning_region_from_country(basin: str, country: str, lat_value: float, lon_value: float) -> str:
    central_america = {
        "Belize",
        "Guatemala",
        "Honduras",
        "El Salvador",
        "Nicaragua",
        "Costa Rica",
        "Panama",
    }
    caribbean = {
        "Cuba",
        "Jamaica",
        "Haiti",
        "Dominican Republic",
        "Bahamas",
        "The Bahamas",
        "Trinidad and Tobago",
        "Barbados",
        "Saint Lucia",
        "Grenada",
        "Dominica",
        "Antigua and Barbuda",
        "Saint Kitts and Nevis",
        "Saint Vincent and the Grenadines",
    }
    if basin == "NA" and -90 <= lon_value <= -55 and 5 <= lat_value <= 24 and country != "Mexico":
        return "Caribbean and Central America"
    if country == "United States of America" and basin == "NA" and lat_value > 24:
        return "United States Gulf-Atlantic"
    if country == "Mexico":
        return "Mexico Pacific" if basin == "EP" else "Mexico Gulf-Caribbean"
    if country in central_america:
        return "Central America Pacific" if basin == "EP" else "Caribbean and Central America"
    if country in caribbean and basin == "NA":
        return "Caribbean and Central America"
    return "Other EPac" if basin == "EP" else "Other Atlantic"


def assign_geospatial_regions(storms: pd.DataFrame) -> pd.DataFrame:
    if storms.empty or not NATURAL_EARTH_COUNTRIES.exists():
        return storms
    points = gpd.GeoDataFrame(
        storms[["storm_id", "basin", "min_approach_lon", "min_approach_lat"]].copy(),
        geometry=[
            Point(xy)
            for xy in zip(storms["min_approach_lon"].to_numpy(dtype=float), storms["min_approach_lat"].to_numpy(dtype=float))
        ],
        crs="EPSG:4326",
    )
    countries = gpd.read_file(NATURAL_EARTH_COUNTRIES)[["ADMIN", "geometry"]].to_crs(ALBERS_NORTH_AMERICA)
    points_proj = points.to_crs(ALBERS_NORTH_AMERICA)
    nearest = gpd.sjoin_nearest(points_proj, countries, how="left", distance_col="nearest_country_m")
    nearest = nearest.sort_values("nearest_country_m").groupby(level=0).head(1).sort_index()
    regions = []
    countries_out = []
    distances_km = []
    for idx, row in nearest.iterrows():
        country = str(row.get("ADMIN", ""))
        basin = str(storms.loc[idx, "basin"])
        lat_value = float(storms.loc[idx, "min_approach_lat"])
        lon_value = float(storms.loc[idx, "min_approach_lon"])
        fallback = str(storms.loc[idx, "coastal_region"])
        distance_km = float(row.get("nearest_country_m", np.nan)) / 1000.0
        countries_out.append(country)
        distances_km.append(distance_km)
        if np.isfinite(distance_km) and distance_km <= 350:
            regions.append(planning_region_from_country(basin, country, lat_value, lon_value))
        else:
            regions.append(fallback)
    out = storms.copy()
    out["nearest_country"] = countries_out
    out["nearest_country_km"] = distances_km
    out["coastal_region"] = regions
    return out


@dataclass
class StormRecord:
    storm_id: str
    name: str
    season: int
    basin: str
    basin_lmi: str
    n_obs_active: int
    genesis_time: str
    lmi_time: str
    dissipation_time: str
    phi_gen: float
    lon_gen: float
    phi_lmi: float
    lon_lmi: float
    phi_dis: float
    lon_dis: float
    vmax_lifetime: float
    pres_at_lmi: float
    duration_h: float
    dmin_point_km: float
    dmin_landfall_km: float
    dmin_coast_km: float
    min_approach_time: str
    min_approach_lat: float
    min_approach_lon: float
    vmax_at_min_approach: float
    category_at_min_approach: str
    translation_speed_min_kmh: float
    coastal_region: str
    ace_6h: float
    ace_per_day: float
    pdi_6h_kt3: float
    coastal_ace_200km: float
    ri_24h_before_min_approach: bool
    ri_48h_before_min_approach: bool
    landfall_ibtracs: bool
    approach_50km: bool
    approach_100km: bool
    approach_200km: bool
    approach_300km: bool
    intensity_class: str
    decade: str
    record_period: str


def parse_times(raw_iso: np.ndarray) -> pd.Series:
    text = pd.Series(decode_array(raw_iso).astype(str))
    text = text.replace({"": pd.NA, "NaT": pd.NA})
    return pd.to_datetime(text, errors="coerce", utc=False)


def first_valid(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    primary = clean_numeric(values)
    if fallback is None:
        return primary
    other = clean_numeric(fallback)
    return np.where(np.isfinite(primary), primary, other)


def storm_intensity_class(vmax: float) -> str:
    if vmax >= INTENSITY_CLASSES["major_plus"]:
        return "major_plus"
    if vmax >= INTENSITY_CLASSES["hu_plus"]:
        return "hu_plus"
    return "ts_plus"


def build_panels(
    ibtracs_path: Path,
    start_year: int,
    end_year: int,
    include_observations: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ibtracs_path.exists():
        raise FileNotFoundError(f"IBTrACS file not found: {ibtracs_path}")

    ds = xr.open_dataset(ibtracs_path, decode_times=False)

    season = clean_numeric(ds["season"].values).astype(float)
    storm_ids = decode_array(ds["sid"].values)
    names = decode_array(ds["name"].values)

    basins = decode_array(ds["basin"].values)
    iso_time = ds["iso_time"].values
    lat = clean_numeric(ds["lat"].values)
    lon = normalize_lon(ds["lon"].values)
    wind = first_valid(ds["usa_wind"].values, ds["wmo_wind"].values)
    pres = first_valid(ds["usa_pres"].values, ds["wmo_pres"].values)
    dist2land = clean_numeric(ds["dist2land"].values)
    landfall = clean_numeric(ds["landfall"].values)
    status = decode_array(ds["usa_status"].values)
    nature = decode_array(ds["nature"].values)

    storm_records: list[dict[str, object]] = []
    obs_records: list[dict[str, object]] = []

    for i in range(len(storm_ids)):
        year = season[i]
        if not np.isfinite(year):
            continue
        year_int = int(year)
        if year_int < start_year or year_int > end_year:
            continue

        basin_i = basins[i]
        relevant = np.isin(basin_i, ["NA", "EP"])
        valid_geo = np.isfinite(lat[i]) & np.isfinite(lon[i])
        valid_wind = np.isfinite(wind[i])
        active = relevant & valid_geo & valid_wind & (wind[i] >= 34.0)
        if active.sum() == 0:
            continue

        vmax = float(np.nanmax(np.where(active, wind[i], np.nan)))
        if vmax < 34.0:
            continue

        active_idx = np.where(active)[0]
        lmi_candidates = active_idx[np.where(wind[i, active_idx] == vmax)[0]]
        lmi_idx = int(lmi_candidates[0])
        gen_idx = int(active_idx[0])
        dis_idx = int(active_idx[-1])
        basin_lmi = str(basin_i[lmi_idx])

        times = parse_times(iso_time[i])
        gen_time = times.iloc[gen_idx]
        lmi_time = times.iloc[lmi_idx]
        dis_time = times.iloc[dis_idx]
        duration_h = float((dis_time - gen_time).total_seconds() / 3600.0) if pd.notna(gen_time) and pd.notna(dis_time) else np.nan

        active_dist = dist2land[i, active_idx]
        active_landfall = landfall[i, active_idx]
        active_wind = wind[i, active_idx]
        active_lat = lat[i, active_idx]
        active_lon = lon[i, active_idx]
        dmin_point = float(np.nanmin(active_dist)) if np.isfinite(active_dist).any() else np.nan
        dmin_landfall = float(np.nanmin(active_landfall)) if np.isfinite(active_landfall).any() else np.nan
        active_coast_dist = np.nanmin(np.vstack([active_dist, active_landfall]), axis=0)
        dmin_coast = float(np.nanmin(active_coast_dist)) if np.isfinite(active_coast_dist).any() else np.nan
        min_pos = int(np.nanargmin(active_coast_dist)) if np.isfinite(active_coast_dist).any() else 0
        min_idx = int(active_idx[min_pos])
        min_time = times.iloc[min_idx]

        active_times = times.iloc[active_idx].reset_index(drop=True)
        elapsed_h = active_times.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        seg_km = np.full(len(active_idx), np.nan)
        if len(active_idx) > 1:
            seg_km[1:] = haversine_km(active_lat[:-1], active_lon[:-1], active_lat[1:], active_lon[1:])
        translation_speed = np.where(np.isfinite(elapsed_h) & (elapsed_h > 0), seg_km / elapsed_h, np.nan)
        dt_forward_h = active_times.shift(-1).sub(active_times).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        weight_6h = np.where(np.isfinite(dt_forward_h) & (dt_forward_h > 0) & (dt_forward_h <= 12), dt_forward_h / 6.0, 1.0)
        ace_6h = float(1.0e-4 * np.nansum((active_wind**2) * weight_6h))
        duration_days = duration_h / 24.0 if np.isfinite(duration_h) and duration_h > 0 else np.nan
        ace_per_day = float(ace_6h / duration_days) if np.isfinite(duration_days) and duration_days > 0 else np.nan
        pdi_6h = float(np.nansum((active_wind**3) * weight_6h))
        coastal_mask_200 = np.isfinite(active_coast_dist) & (active_coast_dist <= 200.0)
        coastal_ace_200 = float(1.0e-4 * np.nansum((active_wind[coastal_mask_200] ** 2) * weight_6h[coastal_mask_200]))

        speed_min = float(translation_speed[min_pos]) if min_pos < len(translation_speed) and np.isfinite(translation_speed[min_pos]) else np.nan
        vmax_min = float(wind[i, min_idx]) if np.isfinite(wind[i, min_idx]) else np.nan
        lat_min = float(lat[i, min_idx]) if np.isfinite(lat[i, min_idx]) else np.nan
        lon_min = float(lon[i, min_idx]) if np.isfinite(lon[i, min_idx]) else np.nan
        region_min = coastal_region(basin_lmi, lat_min, lon_min)

        ri24 = False
        ri48 = False
        if pd.notna(min_time):
            for hours, label in [(24, "ri24"), (48, "ri48")]:
                target = min_time - pd.Timedelta(hours=hours)
                diffs = (active_times - target).abs()
                if diffs.notna().any():
                    prior_pos = int(diffs.argmin())
                    if diffs.iloc[prior_pos] <= pd.Timedelta(hours=3):
                        increase = vmax_min - float(active_wind[prior_pos])
                        if label == "ri24":
                            ri24 = bool(np.isfinite(increase) and increase >= 30.0)
                        else:
                            ri48 = bool(np.isfinite(increase) and increase >= 45.0)

        record = StormRecord(
            storm_id=str(storm_ids[i]),
            name=str(names[i]),
            season=year_int,
            basin=basin_lmi,
            basin_lmi=basin_lmi,
            n_obs_active=int(active.sum()),
            genesis_time="" if pd.isna(gen_time) else gen_time.isoformat(),
            lmi_time="" if pd.isna(lmi_time) else lmi_time.isoformat(),
            dissipation_time="" if pd.isna(dis_time) else dis_time.isoformat(),
            phi_gen=float(lat[i, gen_idx]),
            lon_gen=float(lon[i, gen_idx]),
            phi_lmi=float(lat[i, lmi_idx]),
            lon_lmi=float(lon[i, lmi_idx]),
            phi_dis=float(lat[i, dis_idx]),
            lon_dis=float(lon[i, dis_idx]),
            vmax_lifetime=vmax,
            pres_at_lmi=float(pres[i, lmi_idx]) if np.isfinite(pres[i, lmi_idx]) else np.nan,
            duration_h=duration_h,
            dmin_point_km=dmin_point,
            dmin_landfall_km=dmin_landfall,
            dmin_coast_km=dmin_coast,
            min_approach_time="" if pd.isna(min_time) else min_time.isoformat(),
            min_approach_lat=lat_min,
            min_approach_lon=lon_min,
            vmax_at_min_approach=vmax_min,
            category_at_min_approach=saffir_simpson_category(vmax_min),
            translation_speed_min_kmh=speed_min,
            coastal_region=region_min,
            ace_6h=ace_6h,
            ace_per_day=ace_per_day,
            pdi_6h_kt3=pdi_6h,
            coastal_ace_200km=coastal_ace_200,
            ri_24h_before_min_approach=ri24,
            ri_48h_before_min_approach=ri48,
            landfall_ibtracs=bool(np.isfinite(dmin_landfall) and dmin_landfall <= 0.0),
            approach_50km=bool(np.isfinite(dmin_coast) and dmin_coast <= 50.0),
            approach_100km=bool(np.isfinite(dmin_coast) and dmin_coast <= 100.0),
            approach_200km=bool(np.isfinite(dmin_coast) and dmin_coast <= 200.0),
            approach_300km=bool(np.isfinite(dmin_coast) and dmin_coast <= 300.0),
            intensity_class=storm_intensity_class(vmax),
            decade=decade_label(year_int),
            record_period="common_1949_2025",
        )
        storm_records.append(asdict(record))

        if include_observations:
            for j in active_idx:
                t = times.iloc[int(j)]
                obs_records.append(
                    {
                        "storm_id": str(storm_ids[i]),
                        "name": str(names[i]),
                        "season": year_int,
                        "basin": str(basin_i[j]),
                        "basin_lmi": basin_lmi,
                        "iso_time": "" if pd.isna(t) else t.isoformat(),
                        "lat": float(lat[i, j]),
                        "lon": float(lon[i, j]),
                        "vmax": float(wind[i, j]),
                        "pres": float(pres[i, j]) if np.isfinite(pres[i, j]) else np.nan,
                        "dist2land_km": float(dist2land[i, j]) if np.isfinite(dist2land[i, j]) else np.nan,
                        "landfall_km": float(landfall[i, j]) if np.isfinite(landfall[i, j]) else np.nan,
                        "usa_status": str(status[i, j]),
                        "nature": str(nature[i, j]),
                        "is_lmi": int(j == lmi_idx),
                        "d_obs_km": float(np.nanmin([dist2land[i, j], landfall[i, j]])) if np.isfinite([dist2land[i, j], landfall[i, j]]).any() else np.nan,
                    }
                )

    storms = pd.DataFrame(storm_records)
    storms = assign_geospatial_regions(storms)
    observations = pd.DataFrame(obs_records)
    return storms, observations


def build_annual_panel(storms: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    years = pd.MultiIndex.from_product(
        [["NA", "EP"], range(start_year, end_year + 1)], names=["basin", "season"]
    ).to_frame(index=False)

    grouped = storms.groupby(["basin", "season"], dropna=False)
    annual = grouped.agg(
        n_storms=("storm_id", "count"),
        n_landfall=("landfall_ibtracs", "sum"),
        n_50km=("approach_50km", "sum"),
        n_100km=("approach_100km", "sum"),
        n_200km=("approach_200km", "sum"),
        n_300km=("approach_300km", "sum"),
        n_hurricanes=("vmax_lifetime", lambda x: int((x >= 64).sum())),
        n_major=("vmax_lifetime", lambda x: int((x >= 96).sum())),
        phi_lmi_mean=("phi_lmi", "mean"),
        phi_lmi_median=("phi_lmi", "median"),
        phi_gen_mean=("phi_gen", "mean"),
        dmin_median_km=("dmin_coast_km", "median"),
        vmax_mean=("vmax_lifetime", "mean"),
        ace_6h_sum=("ace_6h", "sum"),
        pdi_6h_sum=("pdi_6h_kt3", "sum"),
        coastal_ace_200km_sum=("coastal_ace_200km", "sum"),
        translation_speed_min_mean=("translation_speed_min_kmh", "mean"),
        n_ri24_pre_min=("ri_24h_before_min_approach", "sum"),
        n_ri48_pre_min=("ri_48h_before_min_approach", "sum"),
    ).reset_index()

    annual = years.merge(annual, on=["basin", "season"], how="left")
    count_cols = ["n_storms", "n_landfall", "n_50km", "n_100km", "n_200km", "n_300km", "n_hurricanes", "n_major"]
    annual[count_cols] = annual[count_cols].fillna(0).astype(int)
    for threshold in THRESHOLDS_KM:
        annual[f"frac_{threshold}km"] = np.where(
            annual["n_storms"] > 0,
            annual[f"n_{threshold}km"] / annual["n_storms"],
            np.nan,
        )
    annual["landfall_frac"] = np.where(annual["n_storms"] > 0, annual["n_landfall"] / annual["n_storms"], np.nan)
    annual["decade"] = annual["season"].map(decade_label)
    return annual


def write_outputs(storms: pd.DataFrame, observations: pd.DataFrame, annual: pd.DataFrame, ibtracs_path: Path) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    storm_csv = PROCESSED / "ibtracs_storm_panel_1949_2025.csv"
    storm_parquet = PROCESSED / "ibtracs_storm_panel_1949_2025.parquet"
    obs_parquet = PROCESSED / "ibtracs_observation_panel_1949_2025.parquet"
    annual_csv = PROCESSED / "annual_coastal_exposure_1949_2025.csv"
    annual_parquet = PROCESSED / "annual_coastal_exposure_1949_2025.parquet"

    storms.to_csv(storm_csv, index=False)
    storms.to_parquet(storm_parquet, index=False)
    if not observations.empty:
        observations.to_parquet(obs_parquet, index=False)
    annual.to_csv(annual_csv, index=False)
    annual.to_parquet(annual_parquet, index=False)

    summary = {
        "source_file": ibtracs_path.name,
        "source_dataset": "IBTrACS version 4r01",
        "source_doi": "10.25921/82ty-9e16",
        "n_storms": int(len(storms)),
        "n_observations": int(len(observations)),
        "season_min": int(storms["season"].min()) if not storms.empty else None,
        "season_max": int(storms["season"].max()) if not storms.empty else None,
        "basin_counts": storms["basin"].value_counts().sort_index().to_dict(),
        "landfall_counts": storms.groupby("basin")["landfall_ibtracs"].sum().astype(int).to_dict(),
        "approach_100km_counts": storms.groupby("basin")["approach_100km"].sum().astype(int).to_dict(),
        "files": {
            "storm_csv": storm_csv.relative_to(ARTICLE).as_posix(),
            "storm_parquet": storm_parquet.relative_to(ARTICLE).as_posix(),
            "observation_parquet": obs_parquet.relative_to(ARTICLE).as_posix() if not observations.empty else None,
            "annual_csv": annual_csv.relative_to(ARTICLE).as_posix(),
            "annual_parquet": annual_parquet.relative_to(ARTICLE).as_posix(),
        },
    }
    (OUTPUTS / "panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibtracs", type=Path, default=DEFAULT_IBTRACS)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END)
    parser.add_argument("--no-observations", action="store_true", help="Skip observation-level output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storms, observations = build_panels(
        ibtracs_path=args.ibtracs,
        start_year=args.start_year,
        end_year=args.end_year,
        include_observations=not args.no_observations,
    )
    annual = build_annual_panel(storms, args.start_year, args.end_year)
    write_outputs(storms, observations, annual, args.ibtracs)
    print(f"Wrote {len(storms):,} storm records and {len(annual):,} basin-year records.")


if __name__ == "__main__":
    main()
