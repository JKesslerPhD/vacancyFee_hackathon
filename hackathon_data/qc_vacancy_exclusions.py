"""
QC: known false-positive categories in vacant_parcels
===============================================================================
Three independent checks against the same list, each catching a different way
a non-vacant (or non-vacancy-tax-eligible) parcel slips into vacant_parcels.csv:

1. Public park/recreation-agency ownership. CLAUDE.md documents Tier 2
   ("Zero Improvement") as excluding parks, but the exclusion isn't actually
   applied upstream: publicly-owned park land is tax-exempt, so it carries $0
   improvement value and reads identically to a vacant lot to the
   zero-improvement heuristic.

2. Tier 2/3 rows whose OWN use code says a structure/active use exists.
   VAL_ASSD_IMPRV is *null* -- not recorded as literally $0 -- for every
   single Tier 2 row: the classification treats missing improvement-value
   data as zero, county-wide, for every use type. Of the ~91 distinct use
   codes remaining in Tier 2/3, the overwhelming majority are specific
   building/business types -- OFFICE BLDG (MULTI-STORY), APARTMENTS,
   HOTEL, RESTAURANT, MEDICAL BLDG/CLINIC, WAREHOUSE, GAS STATION, BANK,
   GOLF COURSE, and dozens more -- not "vacant land" by any description.
   Checking for stray structural fields (BUILDING_SQFT etc.) still isn't
   enough on its own: tax-exempt government parcels routinely have *every*
   field null (55 of 56 "PARKING GARAGE, PARKING STRUCTURE" parcels turned
   out to be actual downtown Sacramento public garages with zero structural
   fields captured at all -- see git history for that narrower, superseded
   version of this check). The robust fix is the other direction: keep an
   explicit allowlist of use codes that actually mean "no structure" --
   VACANT_LAND_USE_CODES below -- and treat everything else in Tier 2/3 as
   miscoded. Tier 1 (land-use-code based, not improvement-value based)
   doesn't have this problem and isn't touched: only 2 of 19,295 Tier 1
   parcels show any structural evidence in the first place.

3. Parking lots, in every tier. Not vacant land -- an operating surface lot
   is paved, in-use commercial property -- and per the campaign's own call,
   not something a vacancy tax would apply to regardless. This empties out
   Tier 3 ("Parking/Abandoned") almost entirely: it turns out to be 100%
   parking lots (BFH-coded) in this dataset, with zero abandoned service
   stations (BFK) actually present despite the tier's name.

All three are downstream QC passes, not fixes to the classification itself -- the
script that builds vacant_parcels.csv (build_hackathon_data.py) isn't in
this repo. Produces a corrected export plus a small audit report of what got
removed and why, so a human can sanity-check the matches before they're
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


# ── Check 2: Tier 2/3 rows whose use code isn't actually "vacant land" ──────
# Only applied to Tier 2/3 -- Tier 1 (land-use-code based) is already clean.
# This is an ALLOWLIST, not a blocklist: anything in Tier 2/3 whose use code
# isn't one of these is treated as miscoded, on the theory that "vacant" use
# codes are a short, enumerable list and "not vacant" use codes are not (see
# module docstring). Reviewed against the full distinct list of what's
# actually in the data -- see qc_exclusion_report.csv after running this.
STRUCTURAL_EVIDENCE_TIERS = {"Tier 2: Zero Improvement", "Tier 3: Parking/Abandoned"}

VACANT_LAND_USE_CODES = {
    "RESIDENTIAL-VACANT LAND",
    "COMMERCIAL-VACANT LAND",
    "INDUSTRIAL-VACANT LAND",
    "RURAL/AGRICULTURAL-VACANT LAND",
    "RECREATIONAL-VACANT LAND",
    "INSTITUTIONAL-VACANT LAND",
    "VACANT LAND (GENERAL)",
    "WASTE LAND, MARSH, SWAMP, SUBMERGED-VACANT LAND",
    "PRIVATE PRESERVE, OPEN SPACE-VACANT LAND (FOREST LAND, CONSERVATION)",
    "QUARRIES (SAND; GRAVEL; ROCK)",       # extraction pit, no structure
    "STORAGE YARD (JUNK; AUTO WRECKING, SALVAGE)",       # outdoor yard, no building
    "STORAGE YARD, OPEN STORAGE (LIGHT EQUIPMENT, MATERIAL)",
    "MISCELLANEOUS (GENERAL)",
}


def is_miscoded_not_vacant_land(row) -> bool:
    if row.get("vacancy_tier") not in STRUCTURAL_EVIDENCE_TIERS:
        return False
    use_code = row.get("USE_CODE_STD_DESC_LPS")
    if not isinstance(use_code, str):
        return False  # missing use code -- not enough to call it miscoded either way
    return use_code not in VACANT_LAND_USE_CODES


# ── Check 3: parking lots -- not a vacancy-tax target, regardless of tier ───
# An operating (or even unstriped/informal) surface parking lot is a paved,
# in-use commercial property, not vacant land, and per the campaign's own
# call isn't something a vacancy tax would apply to. This used to sit in the
# VACANT_LAND_USE_CODES allowlist above (reasoning: "paved, no building") --
# moved to its own check because the exclusion reason is a policy judgment
# about tax eligibility, not a data-quality claim about whether a structure
# exists. Applies across every tier: Tier 3 ("Parking/Abandoned") in this
# dataset turns out to be 100% parking lots (BFH-coded) with zero abandoned
# service stations (BFK) actually present, so this check empties Tier 3 out
# entirely as a side effect, not just trims Tier 2.
def is_parking_lot(row) -> bool:
    return row.get("USE_CODE_STD_DESC_LPS") == "PARKING LOT"


def classify(df: pd.DataFrame) -> pd.Series:
    """Return an exclusion reason per row, or None if not excluded."""
    reasons = pd.Series(None, index=df.index, dtype=object)
    reasons[df["ASSESSEE_OWNER_NAME_1"].apply(is_public_park_owner)] = "public_park_or_recreation_agency"
    # Parking lots before the general miscoded-use-code check: PARKING LOT
    # isn't in VACANT_LAND_USE_CODES, so check 2 would also match it and
    # mislabel it "implies structure" -- it doesn't, the real reason is
    # policy/tax-eligibility (check 3), and that label should win.
    parking = df.apply(is_parking_lot, axis=1)
    reasons[parking & reasons.isna()] = "parking_lot_not_vacancy_tax_eligible"
    miscoded = df.apply(is_miscoded_not_vacant_land, axis=1)
    reasons[miscoded & reasons.isna()] = "use_code_implies_structure_not_vacant_land"
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
