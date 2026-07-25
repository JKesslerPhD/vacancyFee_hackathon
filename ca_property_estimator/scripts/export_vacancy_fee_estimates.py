"""
Adapt CA Property Estimator output for the Vacancy Fee pipeline
===============================================================================
This project's model training runs against the CARB AB 2446 program's
Snowflake warehouse, which vacancyFee_hackathon has no access to and doesn't
need — the estimator was already run once, statewide, producing
results/ca_all_parcels_estimated_values.parquet. This script takes that
pre-computed output, scopes it to Sacramento County, and reshapes it into the
schema the rest of this repo expects (the same schema the retired
parcel_actualValue/ pipeline produced), so results/build_map_data.py and
friends don't need to change.

Inputs:
    results/ca_all_parcels_estimated_values.parquet
        (statewide scored output — not in git, regenerate via src/score_all.py
        against the CARB Snowflake warehouse, or copy from a teammate)
    ../hackathon_data/parcels_trimmed.csv
        (Sacramento assessor data — see hackathon_data/DATA_DOWNLOAD.md)

Output:
    results/parcels_market_value_estimated.csv
        columns: APN, VAL_ASSD, property_type, est_market_value, prop13_benefit,
                 estimation_tier
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

ESTIMATES_PARQUET = PROJECT_DIR / "results" / "ca_all_parcels_estimated_values.parquet"
PARCELS_TRIMMED_CSV = REPO_ROOT / "hackathon_data" / "parcels_trimmed.csv"
OUT_CSV = PROJECT_DIR / "results" / "parcels_market_value_estimated.csv"

# Collapse the estimator's richer PROPERTY_TYPE_GROUP into the 3-bucket scheme
# results/build_map_data.py already groups by (residential / commercial_other
# / vacant) — see build_prop13_simple().
PROPERTY_TYPE_MAP = {
    "residential": "residential",
    "condo": "residential",
    "vacant_land": "vacant",
}


def main() -> None:
    if not ESTIMATES_PARQUET.exists():
        raise SystemExit(
            f"{ESTIMATES_PARQUET} not found — run src/score_all.py against the "
            "CARB Snowflake warehouse first, or copy the file from a teammate "
            "who has."
        )
    if not PARCELS_TRIMMED_CSV.exists():
        raise SystemExit(
            f"{PARCELS_TRIMMED_CSV} not found — see hackathon_data/DATA_DOWNLOAD.md"
        )

    print(f"Loading {ESTIMATES_PARQUET.name}...")
    est = pd.read_parquet(
        ESTIMATES_PARQUET,
        columns=["PARCEL_APN", "COUNTYNAME", "PROPERTY_TYPE_GROUP", "MODEL_TYPE", "ESTIMATED_VALUE"],
    )
    est = est[est["COUNTYNAME"] == "SACRAMENTO"].drop(columns=["COUNTYNAME"])
    est["PARCEL_APN"] = est["PARCEL_APN"].astype(str)
    est = est.drop_duplicates(subset="PARCEL_APN")
    print(f"  {len(est):,} Sacramento County parcels scored")

    print(f"Loading {PARCELS_TRIMMED_CSV.name} for VAL_ASSD...")
    assessed = pd.read_csv(
        PARCELS_TRIMMED_CSV,
        usecols=["PARCEL_APN", "VAL_ASSD"],
        dtype={"PARCEL_APN": str},
        low_memory=False,
    )
    # Sacramento APNs are 14 digits; some rows in parcels_trimmed.csv lost
    # leading zeros upstream (see CLAUDE.md's APN-format note) — repad before
    # joining or ~40% of otherwise-matching rows silently miss.
    assessed["PARCEL_APN"] = assessed["PARCEL_APN"].str.zfill(14)
    assessed = assessed.drop_duplicates(subset="PARCEL_APN")

    df = est.merge(assessed, on="PARCEL_APN", how="left")
    matched = df["VAL_ASSD"].notna().sum()
    print(f"  {matched:,}/{len(df):,} matched to an assessed value ({matched / len(df):.1%})")

    df["property_type"] = df["PROPERTY_TYPE_GROUP"].map(PROPERTY_TYPE_MAP).fillna("commercial_other")
    df["prop13_benefit"] = (df["ESTIMATED_VALUE"] - df["VAL_ASSD"].fillna(0)).clip(lower=0)

    out = df.rename(columns={
        "PARCEL_APN": "APN",
        "ESTIMATED_VALUE": "est_market_value",
        "MODEL_TYPE": "estimation_tier",
    })[["APN", "VAL_ASSD", "property_type", "est_market_value", "prop13_benefit", "estimation_tier"]]

    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(out):,} rows)")
    print(out["property_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
