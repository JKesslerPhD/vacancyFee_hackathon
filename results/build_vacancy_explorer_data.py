"""
Build vacancy_explorer.html — census block group vacancy choropleth +
council district revenue overlay, City of Sacramento only.
===============================================================================
Downloads the Census Bureau's cartographic boundary file for California block
groups (not committed -- see NOTE below on why block groups, not blocks),
clips to city limits, and aggregates revenue_impact's per-parcel output onto
both block groups and council districts.

Run after ca_property_estimator/scripts/export_vacancy_fee_estimates.py and
revenue_impact/estimate_lost_revenue.py.

Outputs (all tracked -- small):
    map_data/block_groups_vacancy.json      -- choropleth: vacant count + revenue per block group
    map_data/council_districts_revenue.json -- district outlines + revenue, for the overlay
    vacancy_explorer.html                   -- vacancy_explorer_template.html with both
                                                JSON files inlined, so the page opens by
                                                double-click (no server, no CORS issue) and
                                                is still a single self-contained file to host
                                                for iframe embedding. Edit the *template*, not
                                                this file directly -- it's overwritten on every run.

NOTE on granularity: true Census blocks (TABBLOCK20) are far too fine for a
citywide choropleth -- Sacramento city has on the order of 10-15K of them,
most single residential blocks with 0-2 vacant parcels, and the statewide
source file is 367MB. Block groups (~400 within city limits, the standard
unit for public data dashboards -- ACS estimates are published at this level)
give a meaningful, fast-loading aggregation instead of a mostly-empty layer.
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
SACRAMENTO_COUNTY_FIPS = "067"

# 5-step validated sequential blue ramp (dataviz skill, references/palette.md,
# steps 250/350/450/550/700) -- passes lightness-monotone, adjacent-gap,
# light-end-contrast, and single-hue checks under `--ordinal`.
CHOROPLETH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
NO_DATA_COLOR = "#e5e3de"

# Minimum block-group area (deg^2, ~= city boundary-clip slivers) to keep.
MIN_AREA_DEG2 = 1e-7


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
    return sac.to_crs("EPSG:4326")


def clip_to_city(block_groups: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    city_boundary = districts.union_all()
    clipped = gpd.clip(block_groups, city_boundary)
    clipped["_area"] = clipped.geometry.area
    clipped = clipped[clipped["_area"] >= MIN_AREA_DEG2].drop(columns="_area")
    print(f"{len(clipped):,} block groups within city limits (slivers below area threshold dropped)")
    return clipped.reset_index(drop=True)


def aggregate_to_block_groups(block_groups: gpd.GeoDataFrame, parcels: pd.DataFrame) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["LONGITUDE"], parcels["LATITUDE"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, block_groups[["GEOID", "geometry"]], how="inner", predicate="within")
    agg = joined.groupby("GEOID").agg(
        vacant_count=("PARCEL_APN", "count"),
        commercial_eligible_count=("commercial_eligible", "sum"),
        total_prop13_gap=("prop13_gap", "sum"),
        total_property_tax_uplift=("potential_property_tax_uplift", "sum"),
        total_sales_tax_total=("estimated_annual_sales_tax_total", "sum"),
        total_sales_tax_city=("estimated_annual_sales_tax_city", "sum"),
    ).reset_index()

    out = block_groups.merge(agg, on="GEOID", how="left")
    for col in ("vacant_count", "commercial_eligible_count", "total_prop13_gap",
                "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city"):
        out[col] = out[col].fillna(0)
    return out


# Metrics the explorer page can color the choropleth by (dropdown toggle).
# Colors for all three are precomputed here so the browser only ever swaps
# which property it reads -- no client-side rebinning.
CHOROPLETH_METRICS = {
    "vacant_count": "fill_color_count",
    "total_property_tax_uplift": "fill_color_proptax",
    "total_sales_tax_total": "fill_color_salestax",
}


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


def assign_choropleth_colors(block_groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = block_groups.copy()
    for value_col, color_col in CHOROPLETH_METRICS.items():
        out[color_col] = _bin_colors(out[value_col])
    return out


def write_block_groups_json(block_groups: gpd.GeoDataFrame) -> str:
    cols = [
        "GEOID", "vacant_count", "commercial_eligible_count", "total_prop13_gap",
        "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city",
        "fill_color_count", "fill_color_proptax", "fill_color_salestax", "geometry",
    ]
    out = block_groups[cols].copy()
    for col in ("vacant_count", "commercial_eligible_count"):
        out[col] = out[col].astype(int)
    for col in ("total_prop13_gap", "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city"):
        out[col] = out[col].round(0).astype(int)

    geojson_str = out.to_json(drop_id=True)
    target = MAP_DATA_DIR / "block_groups_vacancy.json"
    target.write_text(geojson_str)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out):,} block groups, {target.stat().st_size / 1e3:.0f} KB)")
    return geojson_str


def write_districts_json(districts: gpd.GeoDataFrame, district_summary: pd.DataFrame) -> None:
    d = districts.copy()
    d["council_district"] = d.apply(lambda r: f"District {int(r['DISTNUM'])} ({r['NAME']})", axis=1)
    merged = d.merge(district_summary, on="council_district", how="left")

    rename = {
        "vacant_parcels": "vacant_count",
        "commercial_eligible_parcels": "commercial_eligible_count",
        "total_potential_property_tax_uplift": "total_property_tax_uplift",
        "total_estimated_annual_sales_tax_total": "total_sales_tax_total",
        "total_estimated_annual_sales_tax_city": "total_sales_tax_city",
    }
    merged = merged.rename(columns=rename)
    cols = ["DISTNUM", "NAME", "council_district", "vacant_count", "commercial_eligible_count",
            "total_property_tax_uplift", "total_sales_tax_total", "total_sales_tax_city", "geometry"]
    out = merged[cols].copy()
    for col in ("vacant_count", "commercial_eligible_count", "total_property_tax_uplift",
                "total_sales_tax_total", "total_sales_tax_city"):
        out[col] = out[col].round(0).astype(int)

    geojson_str = out.to_json(drop_id=True)
    target = MAP_DATA_DIR / "council_districts_revenue.json"
    target.write_text(geojson_str)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out)} districts, {target.stat().st_size / 1e3:.0f} KB)")
    return geojson_str


def write_html(block_groups_json: str, districts_json: str) -> None:
    template_path = RESULTS_DIR / "vacancy_explorer_template.html"
    html = template_path.read_text()
    html = html.replace("/*__BLOCK_GROUPS_JSON__*/ null", block_groups_json)
    html = html.replace("/*__DISTRICTS_JSON__*/ null", districts_json)

    target = RESULTS_DIR / "vacancy_explorer.html"
    target.write_text(html)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({target.stat().st_size / 1e3:.0f} KB, "
          "data inlined -- opens directly via file://, no server needed)")


def main() -> None:
    if not PARCEL_REVENUE_CSV.exists():
        raise SystemExit(f"{PARCEL_REVENUE_CSV} not found -- run revenue_impact/estimate_lost_revenue.py first")
    if not DISTRICT_SUMMARY_CSV.exists():
        raise SystemExit(f"{DISTRICT_SUMMARY_CSV} not found -- run revenue_impact/estimate_lost_revenue.py first")

    parcels = pd.read_csv(
        PARCEL_REVENUE_CSV,
        usecols=["PARCEL_APN", "LATITUDE", "LONGITUDE", "commercial_eligible", "prop13_gap",
                 "potential_property_tax_uplift", "estimated_annual_sales_tax_total",
                 "estimated_annual_sales_tax_city"],
    )
    districts = gpd.read_file(DISTRICTS_GEOJSON)

    block_groups = fetch_block_groups()
    block_groups = clip_to_city(block_groups, districts)
    block_groups = aggregate_to_block_groups(block_groups, parcels)
    block_groups = assign_choropleth_colors(block_groups)

    MAP_DATA_DIR.mkdir(exist_ok=True)
    block_groups_json = write_block_groups_json(block_groups)

    district_summary = pd.read_csv(DISTRICT_SUMMARY_CSV, index_col=0).reset_index(names="council_district")
    district_summary = district_summary[district_summary["council_district"] != "CITYWIDE TOTAL"]
    districts_json = write_districts_json(districts, district_summary)

    write_html(block_groups_json, districts_json)


if __name__ == "__main__":
    main()
