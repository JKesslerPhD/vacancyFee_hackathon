"""
Build vacancy_explorer.html — property-count choropleth (block groups zoomed
out, census blocks zoomed in) + council district overlay, City of
Sacramento only.
===============================================================================
Downloads Census Bureau boundary files (not committed), clips to city
limits, and aggregates revenue_impact's per-parcel output onto three
geographies: block groups, census blocks, and council districts.

Run after ca_property_estimator/scripts/export_vacancy_fee_estimates.py and
revenue_impact/estimate_lost_revenue.py.

Choropleth metrics: property count, potential property tax uplift, est.
annual sales tax lost, and two "hoarding" proxies -- median years since last
sale (49% of parcels have a recorded sale date) and median market-value/
assessed-value ratio (a Prop 13 proxy for the same thing, available wherever
we have both values: the longer a parcel goes without a reassessment-
triggering sale, the further its market value drifts from its capped
2%/yr-growth assessed base).

Property type: three plain-language buckets (commercial / industrial /
vacant land & residential) -- see property_type_group() in
revenue_impact/estimate_lost_revenue.py -- shown as a breakdown, not a
technical "commercial-eligible" flag.

Outputs:
    map_data/block_groups_vacancy.json      -- tracked, inlined into the HTML
    map_data/council_districts_revenue.json -- tracked, inlined into the HTML
    map_data/blocks_vacancy.json            -- NOT inlined (see below), fetched
                                                lazily when the map is zoomed in
    vacancy_explorer.html                   -- vacancy_explorer_template.html with
                                                block groups + districts + citywide
                                                totals inlined, so the core page opens
                                                by double-click (no server, no CORS
                                                issue). Edit the *template*, not this
                                                file directly -- it's overwritten on
                                                every run.

NOTE on why blocks are fetched, not inlined: 7,442 real census blocks (after
simplification) run ~2.5-3MB of geometry alone -- inlining that would nearly
triple the page's file size for a layer only visible when zoomed in. Instead
it's written as its own JSON and fetched on demand once the map is zoomed
past BLOCK_ZOOM_THRESHOLD (see the template) -- the core experience
(block-group choropleth, district outlines, callout box) still works
offline via file://; the block-level zoom layer needs the page served, same
as it will be once deployed.
"""

from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
MAP_DATA_DIR = RESULTS_DIR / "map_data"

PARCEL_REVENUE_CSV = REPO_ROOT / "revenue_impact" / "results" / "parcel_revenue_estimates.csv"
DISTRICT_SUMMARY_CSV = REPO_ROOT / "revenue_impact" / "results" / "district_revenue_summary.csv"
DISTRICTS_GEOJSON = REPO_ROOT / "maps" / "data" / "council_districts.geojson"

CENSUS_BG_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_06_bg_500k.zip"
CENSUS_BLOCKS_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TABBLOCK20/tl_2023_06_tabblock20.zip"
SACRAMENTO_COUNTY_FIPS = "067"
BLOCK_SIMPLIFY_TOLERANCE_DEG = 0.00003  # ~3m at this latitude

# 5-step validated sequential blue ramp (dataviz skill, references/palette.md,
# steps 250/350/450/550/700) -- passes lightness-monotone, adjacent-gap,
# light-end-contrast, and single-hue checks under `--ordinal`.
CHOROPLETH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
NO_DATA_COLOR = "#e5e3de"

# Minimum clipped-feature area (deg^2, ~= city boundary-clip slivers) to keep.
MIN_AREA_DEG2 = 1e-7

PARCEL_COLS = [
    "PARCEL_APN", "LATITUDE", "LONGITUDE", "property_type_group", "prop13_gap",
    "potential_property_tax_uplift", "estimated_annual_sales_tax_total",
    "estimated_annual_sales_tax_city", "years_since_sale", "market_assessed_ratio",
]


def fetch_block_groups() -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "cb_bg.zip"
        print(f"Downloading {CENSUS_BG_URL} ...")
        urllib.request.urlretrieve(CENSUS_BG_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        shp = next(Path(tmp).glob("*.shp"))
        bg = gpd.read_file(shp)
    sac = bg[(bg["STATEFP"] == "06") & (bg["COUNTYFP"] == SACRAMENTO_COUNTY_FIPS)]
    return sac.rename(columns={"GEOID": "GEOID"}).to_crs("EPSG:4326")


def fetch_census_blocks() -> gpd.GeoDataFrame:
    """Real TABBLOCK20 blocks -- 367MB statewide, no county-scoped download
    exists, so fetch the whole file and filter locally (one-time per run)."""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "tabblock20.zip"
        print(f"Downloading {CENSUS_BLOCKS_URL} (statewide, ~350MB -- filtered to Sacramento County after) ...")
        urllib.request.urlretrieve(CENSUS_BLOCKS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        shp = next(Path(tmp).glob("*.shp"))
        blk = gpd.read_file(shp, columns=["STATEFP20", "COUNTYFP20", "GEOID20", "geometry"])
    sac = blk[blk["COUNTYFP20"] == SACRAMENTO_COUNTY_FIPS].rename(columns={"GEOID20": "GEOID"})
    return sac.to_crs("EPSG:4326")


def clip_to_city(features: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    city_boundary = districts.union_all()
    clipped = gpd.clip(features, city_boundary)
    clipped["_area"] = clipped.geometry.area
    clipped = clipped[clipped["_area"] >= MIN_AREA_DEG2].drop(columns="_area")
    print(f"{len(clipped):,} {label} within city limits (slivers below area threshold dropped)")
    return clipped.reset_index(drop=True)


# Metrics the explorer page can color the choropleth by (dropdown toggle).
# Colors for all five are precomputed here so the browser only ever swaps
# which property it reads -- no client-side rebinning.
CHOROPLETH_METRICS = {
    "property_count": "fill_color_count",
    "total_property_tax_uplift": "fill_color_proptax",
    "total_sales_tax_total": "fill_color_salestax",
    "median_years_since_sale": "fill_color_yrssale",
    "median_market_assessed_ratio": "fill_color_ratio",
}


def aggregate_to_geography(areas: gpd.GeoDataFrame, parcels: pd.DataFrame) -> gpd.GeoDataFrame:
    """Join parcels to whatever polygon layer is passed (block groups or
    blocks -- both have a GEOID column) and aggregate. Shared by both
    geographies so they can't drift out of sync with each other."""
    points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["LONGITUDE"], parcels["LATITUDE"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, areas[["GEOID", "geometry"]], how="inner", predicate="within")
    joined["is_commercial"] = joined["property_type_group"] == "commercial"
    joined["is_industrial"] = joined["property_type_group"] == "industrial"
    joined["is_vacant_res"] = joined["property_type_group"] == "vacant_land_residential"

    agg = joined.groupby("GEOID").agg(
        property_count=("PARCEL_APN", "count"),
        n_commercial=("is_commercial", "sum"),
        n_industrial=("is_industrial", "sum"),
        n_vacant_land_residential=("is_vacant_res", "sum"),
        total_prop13_gap=("prop13_gap", "sum"),
        total_property_tax_uplift=("potential_property_tax_uplift", "sum"),
        total_sales_tax_total=("estimated_annual_sales_tax_total", "sum"),
        total_sales_tax_city=("estimated_annual_sales_tax_city", "sum"),
        # Hoarding signals -- medians, not sums (these are per-parcel rates/
        # durations, not additive dollars). NaN parcels (no recorded sale
        # date, or no assessed value to form a ratio) drop out of the median
        # rather than pulling it toward zero.
        median_years_since_sale=("years_since_sale", "median"),
        median_market_assessed_ratio=("market_assessed_ratio", "median"),
        n_with_sale_date=("years_since_sale", "count"),
    ).reset_index()

    out = areas.merge(agg, on="GEOID", how="left")
    for col in ("property_count", "n_commercial", "n_industrial", "n_vacant_land_residential",
                "total_prop13_gap", "total_property_tax_uplift", "total_sales_tax_total",
                "total_sales_tax_city", "n_with_sale_date"):
        out[col] = out[col].fillna(0)
    return out


def _bin_colors(values: pd.Series) -> pd.Series:
    nonzero = values > 0
    colors = pd.Series(NO_DATA_COLOR, index=values.index)
    if nonzero.sum() >= len(CHOROPLETH_RAMP):
        bins = pd.qcut(values[nonzero], q=len(CHOROPLETH_RAMP), duplicates="drop")
        codes = bins.cat.codes
        colors[nonzero] = [CHOROPLETH_RAMP[min(c, len(CHOROPLETH_RAMP) - 1)] for c in codes]
    elif nonzero.any():
        colors[nonzero] = CHOROPLETH_RAMP[-1]
    return colors


def assign_choropleth_colors(areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = areas.copy()
    for value_col, color_col in CHOROPLETH_METRICS.items():
        out[color_col] = _bin_colors(out[value_col])
    return out


OUT_COLS = [
    "GEOID", "property_count", "n_commercial", "n_industrial", "n_vacant_land_residential",
    "total_prop13_gap", "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city",
    "median_years_since_sale", "median_market_assessed_ratio", "n_with_sale_date",
    "fill_color_count", "fill_color_proptax", "fill_color_salestax",
    "fill_color_yrssale", "fill_color_ratio", "geometry",
]


def _finalize(areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = areas[OUT_COLS].copy()
    for col in ("property_count", "n_commercial", "n_industrial", "n_vacant_land_residential", "n_with_sale_date"):
        out[col] = out[col].astype(int)
    for col in ("total_prop13_gap", "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city"):
        out[col] = out[col].round(0).astype(int)
    for col in ("median_years_since_sale", "median_market_assessed_ratio"):
        out[col] = out[col].round(1)
    return out


def write_block_groups_json(block_groups: gpd.GeoDataFrame) -> str:
    out = _finalize(block_groups)
    geojson_str = out.to_json(drop_id=True)
    target = MAP_DATA_DIR / "block_groups_vacancy.json"
    target.write_text(geojson_str)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out):,} block groups, {target.stat().st_size / 1e3:.0f} KB)")
    return geojson_str


def write_blocks_json(blocks: gpd.GeoDataFrame) -> None:
    simplified = blocks.copy()
    simplified["geometry"] = simplified.geometry.simplify(BLOCK_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    out = _finalize(simplified)
    target = MAP_DATA_DIR / "blocks_vacancy.json"
    target.write_text(out.to_json(drop_id=True))
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out):,} blocks, {target.stat().st_size / 1e6:.1f} MB) -- "
          "NOT inlined, fetched lazily on zoom-in")


def write_districts_json(districts: gpd.GeoDataFrame, district_summary: pd.DataFrame) -> str:
    d = districts.copy()
    d["council_district"] = d.apply(lambda r: f"District {int(r['DISTNUM'])} ({r['NAME']})", axis=1)
    merged = d.merge(district_summary, on="council_district", how="left")

    rename = {
        "vacant_parcels": "property_count",
        "n_commercial": "n_commercial",
        "n_industrial": "n_industrial",
        "n_vacant_land_residential": "n_vacant_land_residential",
        "total_potential_property_tax_uplift": "total_property_tax_uplift",
        "total_estimated_annual_sales_tax_total": "total_sales_tax_total",
        "total_estimated_annual_sales_tax_city": "total_sales_tax_city",
    }
    merged = merged.rename(columns=rename)
    cols = ["DISTNUM", "NAME", "council_district", "property_count",
            "n_commercial", "n_industrial", "n_vacant_land_residential",
            "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city", "geometry"]
    out = merged[cols].copy()
    for col in cols[3:-1]:
        out[col] = out[col].round(0).astype(int)

    geojson_str = out.to_json(drop_id=True)
    target = MAP_DATA_DIR / "council_districts_revenue.json"
    target.write_text(geojson_str)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out)} districts, {target.stat().st_size / 1e3:.0f} KB)")
    return geojson_str


def build_district_summary_with_types(parcels: pd.DataFrame, revenue_summary: pd.DataFrame) -> pd.DataFrame:
    """district_revenue_summary.csv doesn't carry the property-type
    breakdown (added after that script was written) -- compute it here
    straight from the per-parcel data instead of re-plumbing it through
    estimate_lost_revenue.py's own district summary."""
    type_counts = (
        parcels.groupby(["council_district", "property_type_group"]).size().unstack(fill_value=0)
    )
    for col in ("commercial", "industrial", "vacant_land_residential"):
        if col not in type_counts.columns:
            type_counts[col] = 0
    type_counts = type_counts.rename(columns={
        "commercial": "n_commercial", "industrial": "n_industrial",
        "vacant_land_residential": "n_vacant_land_residential",
    }).reset_index()
    return revenue_summary.merge(type_counts, on="council_district", how="left")


def build_citywide_totals(district_summary: pd.DataFrame) -> str:
    """Small JSON blob for the map's callout box -- citywide, city-limits only."""
    totals = district_summary[
        ["vacant_parcels", "n_commercial", "n_industrial", "n_vacant_land_residential",
         "total_potential_property_tax_uplift", "total_estimated_annual_sales_tax_total",
         "total_estimated_annual_sales_tax_city"]
    ].sum()
    payload = {
        "property_count": int(totals["vacant_parcels"]),
        "n_commercial": int(totals["n_commercial"]),
        "n_industrial": int(totals["n_industrial"]),
        "n_vacant_land_residential": int(totals["n_vacant_land_residential"]),
        "total_property_tax_uplift": int(round(totals["total_potential_property_tax_uplift"])),
        "total_sales_tax_total": int(round(totals["total_estimated_annual_sales_tax_total"])),
        "total_sales_tax_city": int(round(totals["total_estimated_annual_sales_tax_city"])),
    }
    return pd.Series(payload).to_json()


def write_html(block_groups_json: str, districts_json: str, citywide_json: str) -> None:
    template_path = RESULTS_DIR / "vacancy_explorer_template.html"
    html = template_path.read_text()
    html = html.replace("/*__BLOCK_GROUPS_JSON__*/ null", block_groups_json)
    html = html.replace("/*__DISTRICTS_JSON__*/ null", districts_json)
    html = html.replace("/*__CITYWIDE_JSON__*/ null", citywide_json)

    target = RESULTS_DIR / "vacancy_explorer.html"
    target.write_text(html)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({target.stat().st_size / 1e3:.0f} KB, "
          "block groups/districts inlined -- opens directly via file://, no server needed)")


def main() -> None:
    if not PARCEL_REVENUE_CSV.exists():
        raise SystemExit(f"{PARCEL_REVENUE_CSV} not found -- run revenue_impact/estimate_lost_revenue.py first")
    if not DISTRICT_SUMMARY_CSV.exists():
        raise SystemExit(f"{DISTRICT_SUMMARY_CSV} not found -- run revenue_impact/estimate_lost_revenue.py first")

    parcels = pd.read_csv(PARCEL_REVENUE_CSV, usecols=PARCEL_COLS + ["DISTNUM"])
    districts = gpd.read_file(DISTRICTS_GEOJSON)

    MAP_DATA_DIR.mkdir(exist_ok=True)

    block_groups = fetch_block_groups()
    block_groups = clip_to_city(block_groups, districts, "block groups")
    block_groups = aggregate_to_geography(block_groups, parcels)
    block_groups = assign_choropleth_colors(block_groups)
    block_groups_json = write_block_groups_json(block_groups)

    blocks = fetch_census_blocks()
    blocks = clip_to_city(blocks, districts, "census blocks")
    blocks = aggregate_to_geography(blocks, parcels)
    blocks = assign_choropleth_colors(blocks)
    write_blocks_json(blocks)

    revenue_summary = pd.read_csv(DISTRICT_SUMMARY_CSV, index_col=0).reset_index(names="council_district")
    revenue_summary = revenue_summary[revenue_summary["council_district"] != "CITYWIDE TOTAL"]
    parcels_with_district = pd.read_csv(PARCEL_REVENUE_CSV, usecols=["council_district", "property_type_group"])
    district_summary = build_district_summary_with_types(parcels_with_district, revenue_summary)

    districts_json = write_districts_json(districts, district_summary)
    citywide_json = build_citywide_totals(district_summary)

    write_html(block_groups_json, districts_json, citywide_json)


if __name__ == "__main__":
    main()
