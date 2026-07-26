"""
Export a human-readable CSV of every vacant parcel in one council district.
===============================================================================
Built to sanity-check the vacancy_tier / use-code labeling by hand, parcel by
parcel -- exactly the audit that caught the Tier 2 "use code says there's a
building here" problem qc_vacancy_exclusions.py now filters out (see that
script's docstring). Run after estimate_lost_revenue.py.

Usage:
    python revenue_impact/export_district_csv.py 4
    python revenue_impact/export_district_csv.py 4 --out ~/Desktop/district4_vacant.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
VACANT_CSV = REPO_ROOT / "hackathon_data" / "vacant_parcels_qc.csv"
PARCEL_REVENUE_CSV = REPO_ROOT / "revenue_impact" / "results" / "parcel_revenue_estimates.csv"

COLUMNS = [
    "PARCEL_APN", "SITE_ADDR", "SITE_CITY", "SITE_ZIP",
    "vacancy_tier", "USE_CODE_STD_DESC_LPS", "USE_CODE_MUNI_DESC", "ZONING",
    "commercial_eligible",
    "ASSESSEE_OWNER_NAME_1",
    "LOT_SIZE_AREA", "BUILDING_SQFT", "YR_BLT",
    "VAL_ASSD", "est_market_value", "prop13_gap",
    "potential_property_tax_uplift", "estimated_annual_sales_tax_total",
    "years_since_sale", "market_assessed_ratio",
    "LATITUDE", "LONGITUDE",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("district", type=int, help="Council district number (1-8)")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = p.parse_args()

    if not PARCEL_REVENUE_CSV.exists():
        raise SystemExit(f"{PARCEL_REVENUE_CSV} not found -- run estimate_lost_revenue.py first")
    if not VACANT_CSV.exists():
        raise SystemExit(f"{VACANT_CSV} not found -- run hackathon_data/qc_vacancy_exclusions.py first")

    revenue = pd.read_csv(PARCEL_REVENUE_CSV, dtype={"PARCEL_APN": str})
    d = revenue[revenue["DISTNUM"] == args.district].copy()
    if d.empty:
        raise SystemExit(f"No parcels found for district {args.district} -- valid range is 1-8")

    vacant = pd.read_csv(
        VACANT_CSV,
        usecols=["PARCEL_APN", "SITE_ADDR", "SITE_CITY", "SITE_ZIP", "USE_CODE_MUNI_DESC",
                 "ZONING", "ASSESSEE_OWNER_NAME_1", "BUILDING_SQFT", "YR_BLT"],
        dtype={"PARCEL_APN": str},
        low_memory=False,
    )
    vacant["PARCEL_APN"] = vacant["PARCEL_APN"].str.zfill(14)

    out = d.merge(vacant, on="PARCEL_APN", how="left")[COLUMNS].sort_values(
        ["vacancy_tier", "USE_CODE_STD_DESC_LPS", "SITE_ADDR"]
    )

    target = args.out or (REPO_ROOT / "revenue_impact" / "results" / f"district{args.district}_vacant_parcels.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)

    print(f"District {args.district}: {len(out):,} vacant parcels -> {target}")
    print()
    print(out["vacancy_tier"].value_counts().to_string())
    print()
    print("Top use codes:")
    print(out["USE_CODE_STD_DESC_LPS"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
