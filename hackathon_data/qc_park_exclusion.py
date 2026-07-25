"""
QC: flag parcels owned by public park/recreation agencies in vacant_parcels
===============================================================================
CLAUDE.md documents Tier 2 ("Zero Improvement") as excluding parks, but the
exclusion isn't actually applied upstream: publicly-owned park land is
tax-exempt, so it carries $0 improvement value and reads identically to a
vacant lot to the zero-improvement heuristic. Coded-vacant land use (Tier 1)
has the same blind spot for undeveloped park/open-space parcels.

This is a downstream QC pass, not a fix to the classification itself — the
script that builds vacant_parcels.csv (build_hackathon_data.py) isn't in this
repo. It owner-name-matches against known public park/recreation agencies and
produces a corrected export plus a small audit report of what got removed, so
a human can sanity-check the matches before they're relied on publicly.

Usage:
    python hackathon_data/qc_park_exclusion.py

Inputs (expected in this directory, per DATA_DOWNLOAD.md):
    vacant_parcels.csv
    vacant_parcels.geojson   (optional — skipped if absent)

Outputs:
    qc_park_exclusion_report.csv   — the flagged rows, for manual review (tracked in git)
    vacant_parcels_qc.csv          — vacant_parcels.csv minus flagged rows
    vacant_parcels_qc.geojson      — vacant_parcels.geojson minus flagged rows (if input present)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HACK_DIR = Path(__file__).resolve().parent
VACANT_CSV = HACK_DIR / "vacant_parcels.csv"
VACANT_GEOJSON = HACK_DIR / "vacant_parcels.geojson"
REPORT_CSV = HACK_DIR / "qc_park_exclusion_report.csv"
CORRECTED_CSV = HACK_DIR / "vacant_parcels_qc.csv"
CORRECTED_GEOJSON = HACK_DIR / "vacant_parcels_qc.geojson"

# Deliberately tight: requires all of PARK + a recreation-department marker +
# a district marker, or an exact state-agency name. Loose substrings like
# "PARK" alone false-positive on private entities (MCCLELLAN BUSINESS PARK
# LLC, PHASE ONE REGIONAL PARK LTD) that aren't park land at all.
_REC_WORD = re.compile(r"\bREC(REATION(AL)?)?\b")
_DISTRICT_WORD = re.compile(r"\bDIST(RICT)?S?\b")
_STATE_AGENCY_PATTERN = re.compile(
    r"^(CA|CALIFORNIA)\s+(DEPT|DEPARTMENT)\s+OF\s+PARKS?\s*(&|AND)\s*RECREATION\b"
    r"|^STATE\s+OF\s+CALIFORNIA\s+PARKS\s*(&|AND)\s*RECREATION\b"
    r"|^CALIFORNIA\s+STATE\s+PARKS\b"
)


def is_public_park_owner(owner_name) -> bool:
    if not isinstance(owner_name, str):
        return False
    o = owner_name.upper()
    if "PARK" in o and _REC_WORD.search(o) and _DISTRICT_WORD.search(o):
        return True
    return bool(_STATE_AGENCY_PATTERN.search(o))


def main() -> None:
    if not VACANT_CSV.exists():
        raise SystemExit(
            f"{VACANT_CSV} not found — download it first "
            "(see DATA_DOWNLOAD.md)."
        )

    df = pd.read_csv(VACANT_CSV, dtype={"PARCEL_APN": str, "TAXAPN": str}, low_memory=False)
    flagged = df["ASSESSEE_OWNER_NAME_1"].apply(is_public_park_owner)

    print(f"vacant_parcels.csv: {len(df):,} rows")
    print(f"flagged as public park/recreation-agency owned: {flagged.sum():,}")
    print(flagged[flagged].index.size and df.loc[flagged, "vacancy_tier"].value_counts().to_string())

    df.loc[flagged].to_csv(REPORT_CSV, index=False)
    df.loc[~flagged].to_csv(CORRECTED_CSV, index=False)
    print(f"wrote {REPORT_CSV.name} ({flagged.sum():,} rows)")
    print(f"wrote {CORRECTED_CSV.name} ({(~flagged).sum():,} rows)")

    if VACANT_GEOJSON.exists():
        import geopandas as gpd

        gdf = gpd.read_file(VACANT_GEOJSON)
        gdf_flagged = gdf["ASSESSEE_OWNER_NAME_1"].apply(is_public_park_owner)
        gdf.loc[~gdf_flagged].to_file(CORRECTED_GEOJSON, driver="GeoJSON")
        print(f"wrote {CORRECTED_GEOJSON.name} ({(~gdf_flagged).sum():,} rows)")
    else:
        print(f"{VACANT_GEOJSON.name} not found, skipping geojson export")


if __name__ == "__main__":
    main()
