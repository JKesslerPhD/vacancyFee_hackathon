"""
Unrealized tax base from vacant parcels, by council district
===============================================================================
See METHODOLOGY.md for the full derivation and sources for every rate below.

Two categories, reported separately because they behave completely
differently -- never sum them into one number:

  * Property tax uplift: CONTINGENT on sale/reassessment (Prop 13 caps
    assessed-value growth regardless of vacancy). A one-time potential, not
    an annual loss.
  * Sales tax: a genuine RECURRING annual loss, for vacant parcels zoned/used
    commercial or retail only.

Inputs:
    ../hackathon_data/vacant_parcels_qc.csv
        (park-excluded vacant parcel list -- run
        hackathon_data/qc_park_exclusion.py first if this doesn't exist yet)
    ../ca_property_estimator/results/parcels_market_value_estimated.csv
        (run ca_property_estimator/scripts/export_vacancy_fee_estimates.py first)
    ../maps/data/council_districts.geojson

Outputs:
    results/parcel_revenue_estimates.csv   (per-parcel detail, gitignored)
    results/district_revenue_summary.csv   (per-district rollup, tracked)
    figures/district_revenue_summary.png
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

VACANT_CSV = REPO_ROOT / "hackathon_data" / "vacant_parcels_qc.csv"
ESTIMATES_CSV = REPO_ROOT / "ca_property_estimator" / "results" / "parcels_market_value_estimated.csv"
DISTRICTS_GEOJSON = REPO_ROOT / "maps" / "data" / "council_districts.geojson"

# ── Rates -- see METHODOLOGY.md for sources ──────────────────────────────────
PROPERTY_TAX_RATE = 0.011          # effective combined rate (all taxing entities)
CAP_RATE = 0.065                   # imputes annual rent from est_market_value
OCCUPANCY_COST_RATIO = 0.08        # rent as a share of gross sales -> implies revenue
SALES_TAX_RATE_TOTAL = 0.0875      # combined state+county+city+district
SALES_TAX_RATE_CITY = 0.02         # city-specific share (1% Bradley-Burns + 1% Measure U)

COMMERCIAL_USE_PATTERN = re.compile(r"COMMERCIAL|RETAIL")


def load_parcels() -> pd.DataFrame:
    if not VACANT_CSV.exists():
        raise SystemExit(f"{VACANT_CSV} not found -- run hackathon_data/qc_park_exclusion.py first")
    if not ESTIMATES_CSV.exists():
        raise SystemExit(
            f"{ESTIMATES_CSV} not found -- run "
            "ca_property_estimator/scripts/export_vacancy_fee_estimates.py first"
        )

    vacant = pd.read_csv(
        VACANT_CSV,
        usecols=["PARCEL_APN", "LATITUDE", "LONGITUDE", "VAL_ASSD", "USE_CODE_STD_DESC_LPS", "vacancy_tier"],
        dtype={"PARCEL_APN": str},
        low_memory=False,
    )
    vacant["PARCEL_APN"] = vacant["PARCEL_APN"].str.zfill(14)

    estimates = pd.read_csv(
        ESTIMATES_CSV,
        usecols=["APN", "est_market_value", "prop13_benefit"],
        dtype={"APN": str},
    )
    estimates["APN"] = estimates["APN"].str.zfill(14)

    df = vacant.merge(estimates, left_on="PARCEL_APN", right_on="APN", how="left")
    matched = df["est_market_value"].notna().sum()
    print(f"{len(df):,} vacant parcels; {matched:,} ({matched / len(df):.1%}) matched a market-value estimate")
    return df


def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prop13_gap"] = df["prop13_benefit"].fillna(0).clip(lower=0)
    df["potential_property_tax_uplift"] = df["prop13_gap"] * PROPERTY_TAX_RATE

    is_commercial = df["USE_CODE_STD_DESC_LPS"].fillna("").str.upper().str.contains(COMMERCIAL_USE_PATTERN)
    df["commercial_eligible"] = is_commercial

    imputed_rent = df["est_market_value"].fillna(0) * CAP_RATE
    imputed_revenue = imputed_rent / OCCUPANCY_COST_RATIO
    df["estimated_annual_business_revenue"] = np.where(is_commercial, imputed_revenue, 0.0)
    df["estimated_annual_sales_tax_total"] = df["estimated_annual_business_revenue"] * SALES_TAX_RATE_TOTAL
    df["estimated_annual_sales_tax_city"] = df["estimated_annual_business_revenue"] * SALES_TAX_RATE_CITY

    return df


def assign_council_district(df: pd.DataFrame) -> pd.DataFrame:
    """Join to a council district and drop everything outside city limits.

    This module is scoped to the City of Sacramento only -- council districts
    partition the city, not the county, so a parcel matching none of the 8
    districts is by definition outside city limits (unincorporated county or
    another incorporated city). Dropped here rather than carried through and
    footnoted later.
    """
    districts = gpd.read_file(DISTRICTS_GEOJSON)
    points = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["LONGITUDE"], df["LATITUDE"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, districts[["DISTNUM", "NAME", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"])
    joined["council_district"] = joined.apply(
        lambda r: f"District {int(r['DISTNUM'])} ({r['NAME']})" if pd.notna(r["DISTNUM"]) else None,
        axis=1,
    )
    outside = joined["council_district"].isna().sum()
    joined = joined[joined["council_district"].notna()].copy()
    print(f"{len(joined):,} vacant parcels within Sacramento city limits "
          f"({outside:,} county/other-city parcels excluded from this analysis)")
    return joined


def summarize_by_district(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("council_district")
        .agg(
            vacant_parcels=("PARCEL_APN", "count"),
            commercial_eligible_parcels=("commercial_eligible", "sum"),
            total_prop13_gap=("prop13_gap", "sum"),
            total_potential_property_tax_uplift=("potential_property_tax_uplift", "sum"),
            total_estimated_annual_sales_tax_total=("estimated_annual_sales_tax_total", "sum"),
            total_estimated_annual_sales_tax_city=("estimated_annual_sales_tax_city", "sum"),
        )
        .sort_values("total_potential_property_tax_uplift", ascending=False)
    )
    citywide = summary.sum(numeric_only=True)
    citywide.name = "CITYWIDE TOTAL"
    return pd.concat([summary, citywide.to_frame().T])


def plot_summary(summary: pd.DataFrame) -> None:
    plot_df = summary.drop(index="CITYWIDE TOTAL").sort_values(
        "total_potential_property_tax_uplift", ascending=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor("#fef9f6")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fef9f6")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    y = np.arange(len(plot_df))
    ax1.barh(y, plot_df["total_potential_property_tax_uplift"] / 1e6, color="#13587f")
    ax1.set_yticks(y, plot_df.index, fontsize=9)
    ax1.set_xlabel("$ millions (one-time, contingent on sale/reassessment)")
    ax1.set_title("Potential property tax uplift\nby council district", fontsize=11, loc="left")

    ax2.barh(y, plot_df["total_estimated_annual_sales_tax_total"] / 1e6, color="#c9622b")
    ax2.set_yticks(y, plot_df.index, fontsize=9)
    ax2.set_xlabel("$ millions / year (recurring, commercial-eligible parcels)")
    ax2.set_title("Estimated annual sales tax lost\nby council district", fontsize=11, loc="left")

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    target = FIGURES_DIR / "district_revenue_summary.png"
    plt.savefig(target, dpi=150, bbox_inches="tight", facecolor="#fef9f6")
    plt.close()
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def main() -> None:
    df = load_parcels()
    df = compute_revenue(df)
    df = assign_council_district(df)

    RESULTS_DIR.mkdir(exist_ok=True)
    parcel_out = RESULTS_DIR / "parcel_revenue_estimates.csv"
    df.to_csv(parcel_out, index=False)
    print(f"wrote {parcel_out.relative_to(REPO_ROOT)} ({len(df):,} rows)")

    summary = summarize_by_district(df)
    summary_out = RESULTS_DIR / "district_revenue_summary.csv"
    summary.to_csv(summary_out)
    print(f"wrote {summary_out.relative_to(REPO_ROOT)}")

    plot_summary(summary)

    display = summary[
        [
            "vacant_parcels",
            "commercial_eligible_parcels",
            "total_potential_property_tax_uplift",
            "total_estimated_annual_sales_tax_total",
            "total_estimated_annual_sales_tax_city",
        ]
    ].copy()
    for col in display.columns[:2]:
        display[col] = display[col].map(lambda v: f"{v:,.0f}")
    for col in display.columns[2:]:
        display[col] = display[col].map(lambda v: f"${v:,.0f}")
    print()
    print(display.to_string())


if __name__ == "__main__":
    main()
