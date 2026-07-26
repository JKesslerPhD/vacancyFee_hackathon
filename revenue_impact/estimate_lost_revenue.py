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

Both are computed from a market-value estimate that gets a sanity cap first
(see cap_market_value()) -- ca_property_estimator produces a small number of
wildly implausible outliers (one <1-acre "COMMERCIAL-VACANT LAND" parcel
priced at $276M, a $7,100+/sqft rate against a $32/sqft citywide median) that
without capping dominate the totals: pre-cap, the top 10 of ~900
commercial-eligible parcels accounted for 56% of the entire citywide sales
tax estimate, badly distorting any district-level or block-group comparison.

Inputs:
    ../hackathon_data/vacant_parcels_qc.csv
        (park-excluded vacant parcel list -- run
        hackathon_data/qc_vacancy_exclusions.py first if this doesn't exist yet)
    ../ca_property_estimator/results/parcels_market_value_estimated.csv
        (ca_property_estimator is an external project, CARB AB 2446, not part
        of this repo -- get its output CSV from a teammate and place it here)
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

# Sanity cap on est_market_value -- see module docstring and METHODOLOGY.md.
# $500/sqft sits well above the 90th percentile of the model's own
# non-outlier output (~$105/sqft citywide) and well below the anomalous
# cluster it produces for a subset of parcels (~$7,100+/sqft) -- there's a
# clean gap in the data between those two numbers, so the exact cap value
# isn't sensitive within that range. Falls back to a multiple of assessed
# value for the ~2.6% of parcels missing LOT_SIZE_AREA.
MAX_DOLLAR_PER_SQFT = 500
FALLBACK_ASSESSED_MULTIPLE = 100   # used only when LOT_SIZE_AREA is missing

# Marsh/drainage/submerged parcels: $500/sqft assumes buildable urban land,
# which these explicitly aren't (that's the whole point of the use code) --
# 1,075 citywide have a median est_market_value/VAL_ASSD ratio of 72x and a
# max of 203,000x under the general per-sqft cap, because a large lot size
# times $500/sqft produces a huge ceiling regardless of whether the land is
# a retention basin. Found via a real example: 3497 San Juan Rd, a Natomas
# drainage/detention parcel assessed at $69, was landing at a modeled
# $1.05M. These fall back to the assessed-multiple cap unconditionally,
# same as parcels missing LOT_SIZE_AREA.
NON_BUILDABLE_USE_CODES = {"WASTE LAND, MARSH, SWAMP, SUBMERGED-VACANT LAND"}

COMMERCIAL_USE_PATTERN = re.compile(r"COMMERCIAL|RETAIL")

# Zoning is messy free text -- hundreds of raw variants, overlay/PUD/SPD
# suffixes, some parcels listing multiple zones -- so this matches only the
# first whitespace-separated token's base code, allowing a trailing
# "-SOMETHING" modifier. Deliberately conservative: known commercial base
# zones (C-1..4, general/shopping/limited/community commercial, business
# park, office building) plus the mixed-use family, which explicitly permits
# ground-floor commercial. Left out ambiguous 2-letter codes that could not
# be verified (SPA, HC, MP, DC, TC, AC, ...) -- see METHODOLOGY.md.
ZONING_COMMERCIAL_PATTERN = re.compile(
    r"^(C-[1-4]|GC|SC|LC|CC|BP|OB|CMU|MU|OPMU|OIMU|RMU|DMU|VCMU)(-|\(|$)"
)


def is_zoned_commercial(zoning) -> bool:
    if not isinstance(zoning, str):
        return False
    token = zoning.split()[0] if zoning.split() else ""
    return bool(ZONING_COMMERCIAL_PATTERN.match(token))


INDUSTRIAL_USE_PATTERN = re.compile(r"INDUSTRIAL|WAREHOUSE|QUARR|STORAGE YARD")
ZONING_INDUSTRIAL_PATTERN = re.compile(r"^(M-[12]|LI)(-|\(|$)")


def is_zoned_industrial(zoning) -> bool:
    if not isinstance(zoning, str):
        return False
    token = zoning.split()[0] if zoning.split() else ""
    return bool(ZONING_INDUSTRIAL_PATTERN.match(token))


def property_type_group(row) -> str:
    """Plain-language property type for the public-facing map -- three
    buckets, not a technical "commercial-eligible" flag. Commercial and
    industrial win on either use-code or zoning signal (see
    is_zoned_commercial/is_zoned_industrial); everything left over -- vacant
    residential land, waste/marsh land, agricultural-vacant land -- is a
    single "vacant land / abandoned residential" bucket, since by this point
    in the pipeline (qc_vacancy_exclusions.py already dropped occupied
    structures and parking lots) what's left in that group is either raw
    unimproved land or a residential parcel whose building is either absent
    or too devalued/non-habitable to carry an improvement value.
    """
    use = row.get("USE_CODE_STD_DESC_LPS")
    use = use.upper() if isinstance(use, str) else ""
    zoning = row.get("ZONING")
    if "COMMERCIAL" in use or "RETAIL" in use or is_zoned_commercial(zoning):
        return "commercial"
    if INDUSTRIAL_USE_PATTERN.search(use) or is_zoned_industrial(zoning):
        return "industrial"
    return "vacant_land_residential"


def load_parcels() -> pd.DataFrame:
    if not VACANT_CSV.exists():
        raise SystemExit(f"{VACANT_CSV} not found -- run hackathon_data/qc_vacancy_exclusions.py first")
    if not ESTIMATES_CSV.exists():
        raise SystemExit(
            f"{ESTIMATES_CSV} not found -- ca_property_estimator is an external "
            "project (CARB AB 2446), not part of this repo. Get its output CSV "
            "from a teammate and place it at that path."
        )

    vacant = pd.read_csv(
        VACANT_CSV,
        usecols=["PARCEL_APN", "LATITUDE", "LONGITUDE", "VAL_ASSD", "USE_CODE_STD_DESC_LPS",
                 "vacancy_tier", "LOT_SIZE_AREA", "LAST_SALE_DATE_TRANSFER", "ZONING"],
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


def cap_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize est_market_value against a $/sqft ceiling; see module docstring."""
    df = df.copy()
    df["est_market_value_uncapped"] = df["est_market_value"]

    has_lot = df["LOT_SIZE_AREA"].fillna(0) > 0
    is_non_buildable = df["USE_CODE_STD_DESC_LPS"].isin(NON_BUILDABLE_USE_CODES)
    per_sqft_cap = df["LOT_SIZE_AREA"] * MAX_DOLLAR_PER_SQFT
    assessed_cap = df["VAL_ASSD"].fillna(0) * FALLBACK_ASSESSED_MULTIPLE

    cap = np.where(has_lot & ~is_non_buildable, per_sqft_cap, assessed_cap)
    capped_mask = df["est_market_value"].fillna(0) > cap
    df["est_market_value"] = np.where(capped_mask, cap, df["est_market_value"])

    n_capped = int(capped_mask.sum())
    if n_capped:
        removed = (df.loc[capped_mask, "est_market_value_uncapped"] - df.loc[capped_mask, "est_market_value"]).sum()
        print(f"capped {n_capped:,} parcels with implausible est_market_value "
              f"(removed ${removed / 1e9:.1f}B in phantom valuation)")
    return df


def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Recompute prop13_gap from the capped estimate rather than trusting
    # prop13_benefit from the CA estimator export, which was computed
    # pre-cap.
    df["prop13_gap"] = (df["est_market_value"] - df["VAL_ASSD"].fillna(0)).clip(lower=0)
    df["potential_property_tax_uplift"] = df["prop13_gap"] * PROPERTY_TAX_RATE

    df["property_type_group"] = df.apply(property_type_group, axis=1)

    # Commercial-eligible = assessor use-code says commercial/retail OR the
    # parcel is zoned for it -- a union, not an intersection. The use-code
    # alone misses 587-754 parcels zoned C-1..4/GC/SC/LC/etc. but coded
    # RESIDENTIAL-VACANT LAND or INDUSTRIAL-VACANT LAND by the assessor; see
    # METHODOLOGY.md for the by-hand audit that found this.
    is_commercial_use = df["USE_CODE_STD_DESC_LPS"].fillna("").str.upper().str.contains(COMMERCIAL_USE_PATTERN)
    is_commercial_zone = df["ZONING"].apply(is_zoned_commercial)
    is_commercial = is_commercial_use | is_commercial_zone
    df["commercial_eligible"] = is_commercial
    df["commercial_basis"] = np.select(
        [is_commercial_use & is_commercial_zone, is_commercial_use, is_commercial_zone],
        ["use_code_and_zoning", "use_code_only", "zoning_only"],
        default="not_commercial",
    )

    imputed_rent = df["est_market_value"].fillna(0) * CAP_RATE
    imputed_revenue = imputed_rent / OCCUPANCY_COST_RATIO
    df["estimated_annual_business_revenue"] = np.where(is_commercial, imputed_revenue, 0.0)
    df["estimated_annual_sales_tax_total"] = df["estimated_annual_business_revenue"] * SALES_TAX_RATE_TOTAL
    df["estimated_annual_sales_tax_city"] = df["estimated_annual_business_revenue"] * SALES_TAX_RATE_CITY

    # Hoarding signals -- see the vacancy_explorer choropleth's "hoarding"
    # metrics. Years since last sale is a direct hold-time read (49% of
    # parcels have a recorded sale date); market/assessed ratio is a Prop 13
    # proxy for the same thing (available wherever we have both values) --
    # the longer a parcel goes without a reassessment-triggering sale, the
    # further its market value drifts from its capped 2%/yr assessed base.
    sale_date = pd.to_datetime(df["LAST_SALE_DATE_TRANSFER"], format="%Y%m%d", errors="coerce")
    df["years_since_sale"] = (pd.Timestamp.now() - sale_date).dt.days / 365.25
    assessed = df["VAL_ASSD"].replace(0, np.nan)
    df["market_assessed_ratio"] = df["est_market_value"] / assessed

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
    df = cap_market_value(df)
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
