"""
Fold 311-signal predicted-vacancy candidates into the vacant parcel set
===============================================================================
The three assessor-code tiers (vacancy_tier "Tier 1"/"Tier 2"/"Tier 3", see
qc_vacancy_exclusions.py) only catch vacant LAND and zero-improvement-value
parcels. A building that's fully vacant but still carries a normal assessed
improvement value -- an empty downtown storefront, an empty office floor --
is invisible to that logic: the assessor's tax roll records that a structure
exists and what it's worth, not whether anyone occupies it. K Street in
downtown Sacramento is the clearest example: 213 parcels, only 9 caught by
the coded tiers, despite the corridor's well-known ground-floor vacancy.

311_heatmap/predict_vacancy.py already builds a transparent alternative
signal for exactly this gap -- a parcel's nearby code-enforcement/health &
safety 311 call history (board-ups, business-compliance weeds, junk &
debris, ...) -- and flags "candidate" parcels not already in the coded-vacant
set whose signal strength matches a typical known-vacant parcel. That output
(maps/data/predicted_vacancies.gpkg) previously only fed the standalone
pyQGIS map suite; this script is what makes it visible everywhere else
(revenue_impact/, results/vacancy_explorer.html, district_briefs/) by
appending it to vacant_parcels_qc.csv as its own tier.

This is a noisier signal than the coded tiers -- predict_vacancy.py's own
precision proxy is ~11%, but that number is a self-referential sanity check
(what share of ALL high-scoring parcels, known-vacant plus new candidates
combined, are already coded vacant), not an external validation of the new
candidates specifically -- it can't distinguish "the model found real hidden
vacancies" from "the model has a high false-positive rate," since adding
more candidates mechanically pulls it down either way. There is no
ground-truth occupancy dataset (e.g. a field survey of current tenancy)
available to validate individual candidates against. So Tier 4 is kept as a
distinct, filterable vacancy_tier value with its underlying vac_score
carried through, not silently blended into "confirmed" categories, and
nothing here claims a validated precision rate for it.

What this script DOES filter is scope, not accuracy: the campaign's target
is vacant commercial/industrial buildings and vacant land, not empty
residential dwellings (an empty single-family home or apartment unit isn't
what a vacant-*building* enforcement program is about, regardless of
whether the 311 signal is real). RESIDENTIAL_DWELLING_USE_CODES below drops
candidates whose use code is a residential dwelling type, reviewed against
the full distinct list of what's actually in the data (see module's git
history / the exclusion report this writes). An earlier version of this
script instead applied qc_vacancy_exclusions.py's vacant-*land* allowlist
here, which was wrong: it flagged ~99% of candidates as "miscoded" for
having a building on them at all, which is true of every Tier 4 row by
design (Tier 4 exists to find vacant buildings, not vacant land) and says
nothing about whether the building is residential vs. commercial/industrial.

Usage:
    python hackathon_data/build_predicted_vacancy_tier.py

Inputs:
    vacant_parcels_qc.csv        (run qc_vacancy_exclusions.py first)
    vacant_parcels_qc.geojson    (optional -- skipped if absent)
    parcels_trimmed.csv          (full assessor attributes to join on APN)
    ../maps/data/predicted_vacancies.gpkg  (run 311_heatmap/predict_vacancy.py first)

Outputs (overwritten in place):
    vacant_parcels_qc.csv           -- + rows tagged vacancy_tier="Tier 4: Predicted (311 Signal)"
    vacant_parcels_qc.geojson       -- same, if the geojson input was present
    tier4_residential_exclusions.csv -- candidates dropped as residential dwellings, for review
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

HACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = HACK_DIR.parent

VACANT_CSV = HACK_DIR / "vacant_parcels_qc.csv"
VACANT_GEOJSON = HACK_DIR / "vacant_parcels_qc.geojson"
PARCELS_TRIMMED_CSV = HACK_DIR / "parcels_trimmed.csv"
PREDICTED_GPKG = REPO_ROOT / "maps" / "data" / "predicted_vacancies.gpkg"
RESIDENTIAL_REPORT_CSV = HACK_DIR / "tier4_residential_exclusions.csv"

TIER_LABEL = "Tier 4: Predicted (311 Signal)"

# Residential dwelling use codes -- excluded from Tier 4 because an empty
# home/unit isn't the target of a vacant-*building* enforcement program,
# unlike an empty storefront or office floor. Reviewed against the full
# distinct USE_CODE_STD_DESC_LPS list among candidates in
# maps/data/predicted_vacancies.gpkg (100 distinct values as of 2026-07-26).
# Deliberately does NOT match on the substring "RESIDENTIAL" or "CONDOMINIUM"
# alone: "COMMERCIAL/OFFICE/RESIDENTIAL (MIXED USE)" (ground-floor commercial
# in a mixed building -- exactly the kind of vacancy this program cares
# about) and "CONDOMINIUM (INDUSTRIAL)"/"CONDOMINIUM OFFICES" (condo-owned
# flex/office space, not housing) are deliberately kept.
RESIDENTIAL_DWELLING_USE_CODES = {
    "SINGLE FAMILY RESIDENTIAL",
    "SINGLE FAMILY RESIDENTIAL (ASSUMED)",
    "CONDOMINIUM UNIT (RESIDENTIAL)",
    "MOBILE/MANUFACTURED HOME (REGARDLESS OF LAND OWNERSHIP)",
    "MOBILE HOME PARK, TRAILER PARK",
    "DUPLEX (2 UNITS, ANY COMBINATION)",
    "TRIPLEX (3 UNITS, ANY COMBINATION)",
    "QUADRUPLEX (4 UNITS, ANY COMBINATION)",
    "APARTMENT HOUSE (5+ UNITS)",
    "APARTMENT HOUSE (100+ UNITS)",
    "APARTMENTS (GENERIC)",
    "GARDEN APT, COURT APT (5+ UNITS)",
    "HIGHRISE APARTMENTS",
    "PLANNED UNIT DEVELOPMENT (PUD) (RESIDENTIAL)",
    "RURAL/AGRICULTURAL RESIDENCE",
    "RESIDENTIAL COMMON AREA (CONDO/PUD/ETC.)",
    "ROW HOUSE (RESIDENTIAL)",
    "TOWNHOUSE (RESIDENTIAL)",
    "RESIDENTIAL INCOME (GENERAL) (MULTI-FAMILY)",
}

# Slimmer schema carried by the geojson (see qc_vacancy_exclusions.py) --
# matched here so the appended rows have the same columns as the existing ones.
GEOJSON_ATTR_COLS = [
    "SITE_ADDR", "SITE_CITY", "SITE_ZIP", "USE_CODE_MUNI_DESC",
    "USE_CODE_STD_DESC_LPS", "VAL_ASSD_LAND", "VAL_ASSD_IMPRV", "VAL_ASSD",
    "LOT_SIZE_AREA", "ZONING", "ASSESSEE_OWNER_NAME_1", "vacancy_tier", "vac_score",
]


def main() -> None:
    if not VACANT_CSV.exists():
        raise SystemExit(f"{VACANT_CSV} not found -- run hackathon_data/qc_vacancy_exclusions.py first")
    if not PREDICTED_GPKG.exists():
        raise SystemExit(f"{PREDICTED_GPKG} not found -- run 311_heatmap/predict_vacancy.py first")
    if not PARCELS_TRIMMED_CSV.exists():
        raise SystemExit(f"{PARCELS_TRIMMED_CSV} not found")

    vacant = pd.read_csv(VACANT_CSV, dtype={"PARCEL_APN": str, "TAXAPN": str}, low_memory=False)
    vacant["PARCEL_APN"] = vacant["PARCEL_APN"].str.zfill(14)
    if "vac_score" not in vacant.columns:
        vacant["vac_score"] = pd.NA
    already_confirmed = set(vacant["PARCEL_APN"])
    print(f"{VACANT_CSV.name}: {len(vacant):,} confirmed-vacant rows "
          f"({vacant['vacancy_tier'].value_counts().to_dict()})")

    candidates = gpd.read_file(PREDICTED_GPKG)
    candidates["PARCEL_APN"] = candidates["PARCEL_APN"].astype(str).str.zfill(14)
    n_total_candidates = len(candidates)
    candidates = candidates[~candidates["PARCEL_APN"].isin(already_confirmed)].copy()
    print(f"predicted_vacancies.gpkg: {n_total_candidates:,} candidates, "
          f"{len(candidates):,} not already in a confirmed tier")

    trimmed = pd.read_csv(PARCELS_TRIMMED_CSV, dtype={"PARCEL_APN": str}, low_memory=False)
    trimmed["PARCEL_APN"] = trimmed["PARCEL_APN"].str.zfill(14)

    candidate_apns = set(candidates["PARCEL_APN"])
    new_rows = trimmed[trimmed["PARCEL_APN"].isin(candidate_apns)].copy()

    # A handful of APNs (condo/multi-unit complexes -- one shared parcel APN,
    # one secured-roll row per unit) appear more than once in parcels_trimmed.csv.
    # The 311-signal model scores one parcel (one geometry), so each candidate
    # APN must contribute exactly one row here too, or those complexes would
    # multiply their parcel/revenue count by their unit count (one seen with
    # 317 unit rows under a single APN). Keep the highest-VAL_ASSD_IMPRV row
    # as the representative record for that parcel.
    n_before_dedup = len(new_rows)
    new_rows = (
        new_rows.sort_values("VAL_ASSD_IMPRV", ascending=False, na_position="last")
        .drop_duplicates(subset="PARCEL_APN", keep="first")
    )
    if n_before_dedup != len(new_rows):
        print(f"  collapsed {n_before_dedup - len(new_rows):,} duplicate-APN rows "
              f"(multi-unit parcels) down to one row per APN")
    print(f"matched {len(new_rows):,}/{len(candidates):,} candidates to a "
          f"{PARCELS_TRIMMED_CSV.name} row")

    # Drop residential dwellings -- see module docstring and
    # RESIDENTIAL_DWELLING_USE_CODES above. Applied here, before appending,
    # since these rows never pass through qc_vacancy_exclusions.py's
    # classify() (they don't exist yet when that script runs).
    n_before_filter = len(new_rows)
    is_residential = new_rows["USE_CODE_STD_DESC_LPS"].isin(RESIDENTIAL_DWELLING_USE_CODES)
    dropped = new_rows[is_residential]
    new_rows = new_rows[~is_residential]
    print(f"  {n_before_filter - len(new_rows):,}/{n_before_filter:,} candidates "
          f"dropped as residential dwellings (not the target of a vacant-"
          f"building program)")
    dropped.to_csv(RESIDENTIAL_REPORT_CSV, index=False)
    print(f"  wrote {RESIDENTIAL_REPORT_CSV.name} ({len(dropped):,} rows, for review)")

    score_by_apn = dict(zip(candidates["PARCEL_APN"], candidates["vac_score"]))
    new_rows["vacancy_tier"] = TIER_LABEL
    new_rows["vac_score"] = new_rows["PARCEL_APN"].map(score_by_apn)

    combined = pd.concat([vacant, new_rows], ignore_index=True)
    combined.to_csv(VACANT_CSV, index=False)
    print(f"wrote {VACANT_CSV.name}: {len(vacant):,} -> {len(combined):,} rows "
          f"(+{len(new_rows):,}, tier='{TIER_LABEL}')")

    if VACANT_GEOJSON.exists():
        existing = gpd.read_file(VACANT_GEOJSON)
        existing_apns = set(existing["APN"].astype(str).str.zfill(14))
        # Restrict to APNs that actually survived the allowlist filter above
        # (new_rows) -- candidates alone still includes the ones dropped for
        # implying an occupied structure, and a left-merge against attrs
        # below would otherwise silently leave those rows with null
        # attributes instead of excluding them.
        surviving_apns = set(new_rows["PARCEL_APN"])
        new_geo_candidates = candidates[
            ~candidates["PARCEL_APN"].isin(existing_apns) & candidates["PARCEL_APN"].isin(surviving_apns)
        ].copy()

        attrs = new_rows.rename(columns={"PARCEL_APN": "APN"})[["APN"] + GEOJSON_ATTR_COLS]
        new_geo = new_geo_candidates.rename(columns={"PARCEL_APN": "APN"})[["APN", "geometry"]].merge(
            attrs, on="APN", how="left"
        )
        new_geo = gpd.GeoDataFrame(new_geo, geometry="geometry", crs=existing.crs)

        combined_geo = gpd.GeoDataFrame(
            pd.concat([existing, new_geo], ignore_index=True), geometry="geometry", crs=existing.crs
        )
        combined_geo.to_file(VACANT_GEOJSON, driver="GeoJSON")
        print(f"wrote {VACANT_GEOJSON.name}: {len(existing):,} -> {len(combined_geo):,} rows")
    else:
        print(f"{VACANT_GEOJSON.name} not found, skipping geojson update")


if __name__ == "__main__":
    main()
