"""
QC: known false-positive categories in vacant_parcels
===============================================================================
Three independent, unrelated checks against the same list, each catching a
different way a non-vacant parcel slips into vacant_parcels.csv:

1. Public park/recreation-agency ownership. CLAUDE.md documents Tier 2
   ("Zero Improvement") as excluding parks, but the exclusion isn't actually
   applied upstream: publicly-owned park land is tax-exempt, so it carries $0
   improvement value and reads identically to a vacant lot to the
   zero-improvement heuristic.

2. Occupied structures miscoded zero-improvement. VAL_ASSD_IMPRV is *null*
   (not recorded as literally $0) for every single Tier 2 row -- the
   classification treats missing improvement-value data as zero. 14% of
   Tier 2 (1,255 of 8,976 parcels) show direct structural evidence --
   building sqft, living sqft, bedroom count, or a year-built -- that
   contradicts "zero improvement." The single biggest category is coded
   SINGLE FAMILY RESIDENTIAL (646 of those 1,255); sampled by hand, several
   are recently-sold occupied homes (e.g. a 2017-built 4BR that sold in
   2022) and City of Sacramento Housing Authority-owned public housing built
   2008. Tier 1 (land-use-code based, not improvement-value based) doesn't
   have this problem: only 2 of 19,295 show any structural evidence.

3. Inherently active uses that check 2 still misses, because tax-exempt
   government parcels often have EVERY structural field null (private
   property gets appraised in more detail than exempt property does): 55 of
   56 "PARKING GARAGE, PARKING STRUCTURE" parcels are actual downtown
   Sacramento public parking garages (City/County/State/community-college-
   district owned) with no BUILDING_SQFT captured at all. Same story for
   "AIRPORT & RELATED" (County of Sacramento, on streets literally named
   Earhart Dr / Lindbergh Dr) and "CEMETERY (EXEMPT)". These three use codes
   describe a use that cannot be "vacant" by definition, unlike offices,
   retail, or golf courses, which legitimately can sit empty/closed and are
   exactly what "vacant building" fee policy targets -- deliberately not
   swept in here.

All three are downstream QC passes, not fixes to the classification itself
-- the script that builds vacant_parcels.csv (build_hackathon_data.py) isn't
in this repo. Produces a corrected export plus a small audit report of what
got removed and why, so a human can sanity-check the matches before they're
relied on publicly.

Usage:
    python hackathon_data/qc_vacancy_exclusions.py

Inputs (expected in this directory, per DATA_DOWNLOAD.md):
    vacant_parcels.csv
    vacant_parcels.geojson   (optional — skipped if absent)

Outputs:
    qc_exclusion_report.csv   — the flagged rows + why, for manual review (tracked in git)
    vacant_parcels_qc.csv     — vacant_parcels.csv minus flagged rows
    vacant_parcels_qc.geojson — vacant_parcels.geojson minus flagged rows (if input present)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HACK_DIR = Path(__file__).resolve().parent
VACANT_CSV = HACK_DIR / "vacant_parcels.csv"
VACANT_GEOJSON = HACK_DIR / "vacant_parcels.geojson"
REPORT_CSV = HACK_DIR / "qc_exclusion_report.csv"
CORRECTED_CSV = HACK_DIR / "vacant_parcels_qc.csv"
CORRECTED_GEOJSON = HACK_DIR / "vacant_parcels_qc.geojson"

# ── Check 1: public park / recreation agency ownership ──────────────────────
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


# ── Check 2: occupied structure despite "zero improvement" ──────────────────
# Only applied to Tier 2/3 -- Tier 1 (land-use-code based) is already clean
# (2 of 19,295 show any structural evidence) and this signal doesn't apply to
# how Tier 1 is derived in the first place.
STRUCTURAL_EVIDENCE_TIERS = {"Tier 2: Zero Improvement", "Tier 3: Parking/Abandoned"}


def has_structural_evidence(row) -> bool:
    if row.get("vacancy_tier") not in STRUCTURAL_EVIDENCE_TIERS:
        return False
    for col in ("BUILDING_SQFT", "LIVING_SQFT", "BEDROOMS"):
        val = row.get(col)
        if pd.notna(val) and val > 0:
            return True
    return pd.notna(row.get("YR_BLT"))


# ── Check 3: inherently active uses, regardless of missing field data ───────
# Government-owned parcels are tax-exempt, so BUILDING_SQFT/YR_BLT/etc. are
# often never captured for them (unlike private property) -- 55 of 56
# "PARKING GARAGE, PARKING STRUCTURE" parcels have every structural field
# null and would sail past check 2, despite being active City/County/State/
# community-college-district parking garages (725 7th St, 1000 I St, etc. --
# downtown Sacramento's actual public parking garages). Same story for
# "AIRPORT & RELATED" (County of Sacramento, on streets literally named
# Earhart Dr / Lindbergh Dr -- Sacramento Executive Airport) and "CEMETERY
# (EXEMPT)" (cemetery districts, the U.S. government, the Catholic diocese).
# Deliberately NOT extended to ambiguous categories like offices, retail, or
# golf courses -- those legitimately CAN sit vacant/closed and are exactly
# what "vacant building" fee policy targets; a parking structure, airport, or
# cemetery cannot be "vacant" in that sense by definition of the use itself.
INHERENTLY_ACTIVE_USES = {
    "PARKING GARAGE, PARKING STRUCTURE",
    "AIRPORT & RELATED",
    "CEMETERY (EXEMPT)",
}


def is_inherently_active_use(row) -> bool:
    if row.get("vacancy_tier") not in STRUCTURAL_EVIDENCE_TIERS:
        return False
    return row.get("USE_CODE_STD_DESC_LPS") in INHERENTLY_ACTIVE_USES


def classify(df: pd.DataFrame) -> pd.Series:
    """Return an exclusion reason per row, or None if not excluded."""
    reasons = pd.Series(None, index=df.index, dtype=object)
    reasons[df["ASSESSEE_OWNER_NAME_1"].apply(is_public_park_owner)] = "public_park_or_recreation_agency"
    structural = df.apply(has_structural_evidence, axis=1)
    active_use = df.apply(is_inherently_active_use, axis=1)
    # Don't overwrite a park reason that also happens to show structural
    # evidence (e.g. a park building) -- keep the more specific park reason.
    reasons[structural & reasons.isna()] = "occupied_structure_missing_improvement_data"
    reasons[active_use & reasons.isna()] = "inherently_active_use_parking_airport_cemetery"
    return reasons


def main() -> None:
    if not VACANT_CSV.exists():
        raise SystemExit(
            f"{VACANT_CSV} not found — download it first "
            "(see DATA_DOWNLOAD.md)."
        )

    df = pd.read_csv(VACANT_CSV, dtype={"PARCEL_APN": str, "TAXAPN": str}, low_memory=False)
    # Sacramento APNs are 14 digits; some vacant_parcels.csv rows lost leading
    # zeros upstream (same issue as ca_property_estimator's export script) --
    # repad so the APN join against the geojson below doesn't silently miss.
    df["PARCEL_APN"] = df["PARCEL_APN"].str.zfill(14)
    reasons = classify(df)
    flagged = reasons.notna()

    print(f"vacant_parcels.csv: {len(df):,} rows")
    print(f"flagged for exclusion: {flagged.sum():,}")
    print(reasons.value_counts().to_string())
    print()
    print(pd.crosstab(reasons[flagged], df.loc[flagged, "vacancy_tier"]).to_string())

    report = df.loc[flagged].copy()
    report["exclusion_reason"] = reasons[flagged]
    report.to_csv(REPORT_CSV, index=False)
    df.loc[~flagged].to_csv(CORRECTED_CSV, index=False)
    print(f"\nwrote {REPORT_CSV.name} ({flagged.sum():,} rows)")
    print(f"wrote {CORRECTED_CSV.name} ({(~flagged).sum():,} rows)")

    if VACANT_GEOJSON.exists():
        import geopandas as gpd

        # vacant_parcels.geojson carries a slimmer schema than the CSV --
        # no BUILDING_SQFT/LIVING_SQFT/BEDROOMS/YR_BLT -- so classify() can't
        # run on it directly (has_structural_evidence would silently see
        # every column as missing and never flag anything). Reuse the
        # exclusion set already computed from the full CSV instead, joined
        # on APN, so both outputs agree on exactly the same excluded parcels.
        excluded_apns = set(df.loc[flagged, "PARCEL_APN"])
        gdf = gpd.read_file(VACANT_GEOJSON)
        gdf_flagged = gdf["APN"].astype(str).isin(excluded_apns)
        gdf.loc[~gdf_flagged].to_file(CORRECTED_GEOJSON, driver="GeoJSON")
        print(f"wrote {CORRECTED_GEOJSON.name} ({(~gdf_flagged).sum():,} rows)")
    else:
        print(f"{VACANT_GEOJSON.name} not found, skipping geojson export")


if __name__ == "__main__":
    main()
