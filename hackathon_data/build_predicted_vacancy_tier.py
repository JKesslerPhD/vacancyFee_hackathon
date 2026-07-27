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
precision proxy is ~11% (most high-scoring parcels are false positives, e.g.
a weeds complaint on an occupied business) -- so it's kept as a distinct,
filterable vacancy_tier value with its underlying vac_score carried through,
not silently blended into "confirmed" categories.

That ~11% precision proxy undersold it: before this script applied
qc_vacancy_exclusions.py's own use-code allowlist to candidates before
appending them, 98.7% of Tier 4 rows (6,846 of 6,939) had a use code
implying a real, occupied structure -- department stores, high-rise
apartments, offices, a theater -- because large, busy, occupied buildings
generate plenty of 311 calls for reasons that have nothing to do with
vacancy. This one gap accounted for 77.5% of the citywide sales-tax
estimate and 53.2% of the property-tax uplift estimate downstream. The
allowlist is the same check the coded tiers already get in
qc_vacancy_exclusions.py, applied here to candidates before they're
appended, not after.

Usage:
    python hackathon_data/build_predicted_vacancy_tier.py

Inputs:
    vacant_parcels_qc.csv        (run qc_vacancy_exclusions.py first)
    vacant_parcels_qc.geojson    (optional -- skipped if absent)
    parcels_trimmed.csv          (full assessor attributes to join on APN)
    ../maps/data/predicted_vacancies.gpkg  (run 311_heatmap/predict_vacancy.py first)

Outputs (overwritten in place):
    vacant_parcels_qc.csv       -- + rows tagged vacancy_tier="Tier 4: Predicted (311 Signal)"
    vacant_parcels_qc.geojson   -- same, if the geojson input was present
    tier4_allowlist_report.csv  -- candidates dropped by the use-code allowlist, for review
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from qc_vacancy_exclusions import VACANT_LAND_USE_CODES

HACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = HACK_DIR.parent

VACANT_CSV = HACK_DIR / "vacant_parcels_qc.csv"
VACANT_GEOJSON = HACK_DIR / "vacant_parcels_qc.geojson"
PARCELS_TRIMMED_CSV = HACK_DIR / "parcels_trimmed.csv"
PREDICTED_GPKG = REPO_ROOT / "maps" / "data" / "predicted_vacancies.gpkg"
ALLOWLIST_REPORT_CSV = HACK_DIR / "tier4_allowlist_report.csv"

TIER_LABEL = "Tier 4: Predicted (311 Signal)"

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

    # Same allowlist qc_vacancy_exclusions.py applies to the coded tiers --
    # see module docstring. Applied here, before appending, since these rows
    # never pass through classify() there (they don't exist yet when that
    # script runs).
    n_before_allowlist = len(new_rows)
    fails_allowlist = ~new_rows["USE_CODE_STD_DESC_LPS"].isin(VACANT_LAND_USE_CODES)
    dropped = new_rows[fails_allowlist]
    new_rows = new_rows[~fails_allowlist]
    print(f"  {n_before_allowlist - len(new_rows):,}/{n_before_allowlist:,} candidates "
          f"dropped by the vacant-land-use-code allowlist (use code implies an "
          f"occupied structure, not vacant land)")
    dropped.to_csv(ALLOWLIST_REPORT_CSV, index=False)
    print(f"  wrote {ALLOWLIST_REPORT_CSV.name} ({len(dropped):,} rows, for review)")

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
