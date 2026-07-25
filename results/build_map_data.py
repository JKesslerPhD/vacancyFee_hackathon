"""
Build the public-facing map + figure assets for results/index.html.

Outputs (all under results/map_data/ and results/figures/):
  * map_data/vacant_parcels.json      - compact array of vacant parcel points (no PII)
  * map_data/hs_311.json              - housing / health-&-safety 311 call coordinates
  * map_data/hotspots_summary.json    - city-level rollup for callouts
  * figures/01_prop13_gap_simple.png  - voter-friendly gap chart
  * figures/03_signal_lift_simple.png - top 311 co-occurrences

This is a one-time build script — re-run after pipeline changes.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent
MAPDIR = RESULTS / "map_data"
FIGDIR = RESULTS / "figures"
MAPDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# ── 1. Vacant parcel centroids (PII stripped, compact array format) ─────────
def build_parcel_geojson() -> None:
    # Prefer the QC-corrected export (public park/rec-district parcels
    # removed, see hackathon_data/qc_park_exclusion.py) when present locally.
    vacant_path = ROOT / "hackathon_data" / "vacant_parcels_qc.geojson"
    if not vacant_path.exists():
        vacant_path = ROOT / "hackathon_data" / "vacant_parcels.geojson"
    print(f"Loading {vacant_path.name}...")
    gdf = gpd.read_file(vacant_path)
    print(f"  {len(gdf):,} vacant parcels")

    print("Joining hybrid market estimates...")
    mv = pd.read_csv(
        ROOT / "ca_property_estimator" / "results" / "parcels_market_value_estimated.csv",
        usecols=["APN", "est_market_value", "prop13_benefit", "estimation_tier"],
        dtype={"APN": str},
    ).drop_duplicates(subset="APN")
    gdf["APN"] = gdf["APN"].astype(str)
    gdf = gdf.drop_duplicates(subset="APN").merge(mv, on="APN", how="left")

    # Project to a planar CRS for accurate centroids, then back to lat/lon.
    centroids = gdf.geometry.to_crs("EPSG:3310").centroid.to_crs("EPSG:4326")
    gdf["lat"] = centroids.y.round(5)
    gdf["lon"] = centroids.x.round(5)
    gdf = gdf[gdf["lat"].notna() & gdf["lon"].notna()]

    # Tier code as 1/2/3 instead of "Tier 1: Coded Vacant"
    tier_map = {"Tier 1: Coded Vacant": 1, "Tier 2: Zero Improvement": 2, "Tier 3: Parking/Abandoned": 3}
    gdf["tcode"] = gdf["vacancy_tier"].map(tier_map).fillna(0).astype(int)

    # Categorize use into a single letter for the legend (R / C / V / O)
    use_map = {"R": "Residential", "C": "Commercial", "V": "Vacant land", "I": "Industrial", "O": "Other"}
    def use_class(desc) -> str:
        d = desc.upper() if isinstance(desc, str) else ""
        if "VAC" in d: return "V"
        if "RES" in d or "SFD" in d or "APT" in d or "DUP" in d: return "R"
        if "COM" in d or "OFFICE" in d or "RETAIL" in d or "STORE" in d: return "C"
        if "IND" in d or "WAREHOUSE" in d or "MFG" in d: return "I"
        return "O"
    gdf["uc"] = gdf["USE_CODE_MUNI_DESC"].apply(use_class)

    # Compact array format: each row is [lat, lon, tier, useclass, assessed, market_est, gap, lot_sqft, city, apn]
    cities = sorted(gdf["SITE_CITY"].dropna().unique())
    city_idx = {c: i for i, c in enumerate(cities)}

    def n(x):
        if pd.isna(x): return 0
        try: return int(round(float(x)))
        except Exception: return 0

    points = []
    for r in gdf.itertuples(index=False):
        points.append([
            r.lat, r.lon,
            int(r.tcode),
            r.uc,
            n(r.VAL_ASSD),
            n(r.est_market_value),
            n(r.prop13_benefit),
            n(r.LOT_SIZE_AREA),
            city_idx.get(r.SITE_CITY, -1),
            r.APN,
        ])

    payload = {
        "schema": ["lat", "lon", "tier", "use", "assessed", "market", "gap", "lot_sqft", "city_id", "apn"],
        "tiers":  {"1": "Coded vacant land", "2": "Zero improvement", "3": "Parking/abandoned"},
        "uses":   use_map,
        "cities": cities,
        "points": points,
    }

    target = MAPDIR / "vacant_parcels.json"
    target.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = target.stat().st_size / 1e6
    print(f"  -> {target.name}  ({len(points):,} points, {size_mb:.1f} MB)")


# ── 2. 311 health-&-safety signal points (housing-related calls only) ────────
HOUSING_CATS = (
    "Code Enforcement Housing - Boardup",
    "Code Enforcement Housing - Complaint",
    "Code Enforcement Emergency Housing Repair Program - Complaint",
    "Animal Control Abandoned",
    "Code Enforcement Pest",
    "Code Enforcement Junk & Debris",
)

CAT_LABELS = {
    "Code Enforcement Housing - Boardup": "Boarded-up building",
    "Code Enforcement Housing - Complaint": "Housing code complaint",
    "Code Enforcement Emergency Housing Repair Program - Complaint": "Emergency housing repair",
    "Animal Control Abandoned": "Abandoned animal",
    "Code Enforcement Pest": "Pest infestation",
    "Code Enforcement Junk & Debris": "Junk & debris",
}


def build_311_json() -> None:
    print("Loading housing-related 311 calls...")
    quoted = "(" + ",".join(f"'{c}'" for c in HOUSING_CATS) + ")"
    gdf = gpd.read_file(
        ROOT / "data" / "SacCounty_SalesForce311_calls.gpkg",
        layer="SalesForce311",
        columns=["CategoryName"],
        where=f"CategoryName IN {quoted}",
    )
    print(f"  {len(gdf):,} housing / health-&-safety calls")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf["lat"] = gdf.geometry.y.round(5)
    gdf["lon"] = gdf.geometry.x.round(5)

    label_ids = {cat: i for i, cat in enumerate(HOUSING_CATS)}
    gdf["lid"] = gdf["CategoryName"].map(label_ids)
    payload = {
        "labels": [CAT_LABELS[c] for c in HOUSING_CATS],
        "points": [[float(r.lat), float(r.lon), int(r.lid)]
                   for r in gdf[["lat", "lon", "lid"]].dropna().itertuples(index=False)],
    }

    target = MAPDIR / "hs_311.json"
    target.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = target.stat().st_size / 1e6
    print(f"  -> {target.name}  ({len(payload['points']):,} points, {size_mb:.1f} MB)")


# ── 3. City-level summary for hotspot callouts ───────────────────────────────
def build_hotspot_summary() -> None:
    print("Computing city-level hotspot summary...")
    payload = json.loads((MAPDIR / "vacant_parcels.json").read_text())
    cities = payload["cities"]
    df = pd.DataFrame(payload["points"], columns=payload["schema"])
    df["city"] = df["city_id"].apply(lambda i: cities[i] if 0 <= i < len(cities) else None)
    by_city = (
        df.dropna(subset=["city"])
        .groupby("city")
        .agg(parcels=("apn", "count"), total_market=("market", "sum"), total_gap=("gap", "sum"))
        .sort_values("parcels", ascending=False)
        .head(8)
    )
    out = [
        {
            "city": city,
            "parcels": int(row["parcels"]),
            "market_billions": round(row["total_market"] / 1e9, 2),
            "gap_billions": round(row["total_gap"] / 1e9, 2),
        }
        for city, row in by_city.iterrows()
    ]
    (MAPDIR / "hotspots_summary.json").write_text(json.dumps(out, indent=2))
    print(f"  -> hotspots_summary.json  ({len(out)} cities)")


# ── 4. Simplified Prop 13 gap figure ─────────────────────────────────────────
def build_prop13_simple() -> None:
    print("Building voter-friendly Prop 13 figure...")
    df = pd.read_csv(
        ROOT / "ca_property_estimator" / "results" / "parcels_market_value_estimated.csv",
        usecols=["property_type", "VAL_ASSD", "est_market_value"],
    )
    df = df.dropna(subset=["VAL_ASSD", "est_market_value"])
    df = df[(df["VAL_ASSD"] > 1000) & (df["est_market_value"] > 1000)]

    summary = (
        df.groupby("property_type")
        .agg(assessed=("VAL_ASSD", "median"), market=("est_market_value", "median"))
        .reindex(["residential", "commercial_other", "vacant"])
        .dropna()
    )
    type_labels = {"residential": "Homes", "commercial_other": "Commercial & other", "vacant": "Vacant land"}
    summary.index = [type_labels.get(i, i) for i in summary.index]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#fef9f6")
    ax.set_facecolor("#fef9f6")

    y = np.arange(len(summary))
    h = 0.36
    ax.barh(y - h / 2, summary["assessed"], height=h, color="#b9b3ad", label="Assessed value (what's taxed)")
    ax.barh(y + h / 2, summary["market"], height=h, color="#13587f", label="Market value (what it's worth)")

    for i, (assd, mkt) in enumerate(zip(summary["assessed"], summary["market"])):
        ax.text(assd, i - h / 2, f"  ${assd/1000:,.0f}K", va="center", fontsize=10, color="#15191c")
        ax.text(mkt, i + h / 2, f"  ${mkt/1000:,.0f}K", va="center", fontsize=10, color="#15191c", fontweight="bold")
        gap_pct = (mkt - assd) / assd * 100
        ax.text(
            max(assd, mkt) * 1.18, i,
            f"{gap_pct:+.0f}% gap",
            va="center", fontsize=11, color="#13587f", fontweight="bold",
        )

    ax.set_yticks(y, summary.index, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, summary[["assessed", "market"]].values.max() * 1.5)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_title(
        "Median property value: what Sacramento taxes vs. what it's worth",
        fontsize=13, color="#15191c", pad=14, loc="left",
    )
    plt.tight_layout()
    target = FIGDIR / "01_prop13_gap_simple.png"
    plt.savefig(target, dpi=150, bbox_inches="tight", facecolor="#fef9f6")
    plt.close()
    print(f"  -> {target.name}")


# ── 5. Simplified 311 signal-lift figure ─────────────────────────────────────
def build_signal_lift_simple() -> None:
    """Voter-friendly co-occurrence figure.

    Fix (per Jeff's review of the old version): the bold number on each bar and
    the sentence beneath it now use the SAME value — the actual ``lift`` from
    correlated_pairs.csv — instead of a hand-typed "~6×" approximation that
    disagreed with the bar. "Lift" is stated plainly so the multiplier is
    self-explanatory, and the relationship is phrased symmetrically (two signals
    *share an address* N× more often than chance) because lift is symmetric.
    """
    print("Building voter-friendly 311 signal-lift figure...")
    pairs = pd.read_csv(ROOT / "311_heatmap" / "correlated_pairs.csv")

    # (category_1, category_2, short pair label, plain-English subject of the
    # sentence). The multiplier is read from the data, never hand-written.
    picks = [
        ("Code Enforcement Housing - Boardup", "Homeless Camp - Primary Private Property",
         "Board-ups ↔ encampments",
         "Boarded-up buildings and homeless-camp reports"),
        ("Code Enforcement Housing - Complaint", "Code Enforcement Pest",
         "Housing complaints ↔ pests",
         "Housing-code complaints and pest-control calls"),
        ("Animal Control Abandoned", "Code Enforcement Housing - Complaint",
         "Abandoned animals ↔ housing",
         "Abandoned-animal calls and housing-code complaints"),
        ("Code Enforcement Housing - Boardup", "Code Enforcement Junk & Debris",
         "Board-ups ↔ dumping",
         "Boarded-up buildings and junk & debris dumping"),
    ]

    rows = []
    for c1, c2, label, subject in picks:
        match = pairs[((pairs["category_1"] == c1) & (pairs["category_2"] == c2))
                      | ((pairs["category_1"] == c2) & (pairs["category_2"] == c1))]
        if match.empty:
            continue
        row = match.iloc[0]
        lift = float(row["lift"])
        n = int(row["co_occur_addrs"])
        # Text multiplier derived FROM the data so it always matches the bar.
        blurb = (f"{subject} show up at the same address "
                 f"{lift:.1f}× more often than random chance.")
        rows.append((label, blurb, lift, n))

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor("#fef9f6")
    ax.set_facecolor("#fef9f6")

    y = np.arange(len(rows))
    labels = [r[0] for r in rows]
    blurbs = [r[1] for r in rows]
    lifts = [r[2] for r in rows]
    addrs = [r[3] for r in rows]

    ax.barh(y, lifts, color="#13587f", height=0.5)
    for i, (lift, n) in enumerate(zip(lifts, addrs)):
        ax.text(lift + max(lifts) * 0.02, i,
                f"{lift:.1f}×   ({n:,} shared addresses)",
                va="center", fontsize=11, color="#15191c", fontweight="bold")

    ax.set_yticks(y, labels, fontsize=10.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(lifts) * 1.5)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    # Plain sentence beneath each bar — same multiplier as the bar.
    for i, b in enumerate(blurbs):
        ax.text(0, i + 0.34, b, fontsize=9.5, color="#5a6068", va="top")

    ax.set_title(
        "How health & safety 311 calls cluster together",
        fontsize=14, color="#15191c", pad=34, loc="left",
    )
    ax.text(0, 1.012,
            "“Lift” = how many times more often two complaint types share an "
            "address than random chance (higher = tighter link)",
            transform=ax.transAxes, fontsize=10, color="#5a6068", va="bottom")
    fig.subplots_adjust(top=0.84)
    target = FIGDIR / "03_signal_lift_simple.png"
    plt.savefig(target, dpi=150, bbox_inches="tight", facecolor="#fef9f6")
    plt.close()
    print(f"  -> {target.name}")


def main() -> None:
    build_parcel_geojson()
    build_311_json()
    build_hotspot_summary()
    build_prop13_simple()
    build_signal_lift_simple()
    print("\nDone.")


if __name__ == "__main__":
    main()
