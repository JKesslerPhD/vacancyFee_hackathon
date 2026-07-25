"""
Predict vacancy from a parcel's 311 signal profile
===============================================================================
Turns the vacancy↔311 relationship around. The synthesis script asks "do vacant
parcels draw more health & safety calls?"; this asks the inverse, which is the
framing the campaign wants:

    Vacancy ~ 311 signals  →  use the strongest-correlated 311 categories to
    predict which parcels are (probably) vacant.

This is a deliberately *transparent* model — a weighted signal score, not a
black box:

    1. Attribute each health-&-safety 311 call to its nearest parcel (<=150 ft).
    2. For each 311 category, measure its **vacancy lift**:
           lift_c = P(parcel vacant | parcel has a call of category c)
                    -------------------------------------------------
                                P(parcel vacant)            (base rate)
       A lift of 4 means "a parcel with this call type is 4x more likely to be
       vacant than a random parcel."
    3. Score every parcel:  score = Σ  max(log2(lift_c), 0)  over the categories
       it actually has. Only positively-predictive signals add evidence; the
       weights are interpretable (each is "bits of vacancy evidence").
    4. Flag **candidate vacancies**: parcels NOT already in the coded-vacant set
       whose score is at least as vacancy-like as a typical known-vacant parcel.
       These are the "hidden" vacancies the 311 signal surfaces.

Outputs:
    figures/predicted_vacancy_signals.png   — top signals by vacancy lift
    ../maps/data/predicted_vacancies.gpkg    — candidate parcels (EPSG:4326) for
                                               the pyQGIS predicted-vacancy maps
    appends prediction stats to findings.json

Reuses the loaders (and the APN zero-pad fix) from vacancy_311_synthesis.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vacancy_311_synthesis as S  # loaders, constants, palette, helpers

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FINDINGS = SCRIPT_DIR / "findings.json"
PRED_GPKG = SCRIPT_DIR.parent / "maps" / "data" / "predicted_vacancies.gpkg"

# A category must touch at least this many parcels to earn a weight (avoids a
# huge lift computed from a handful of parcels).
MIN_SUPPORT_PARCELS = 200

# Actively-managed public land -- parks, plazas, government buildings -- draws
# 311 calls (encampments, illegal dumping, code enforcement) *because* it's
# open, accessible public space, not because it's neglected private property.
# The signal model conflates the two: before this filter existed, its top-two
# midtown candidates were Cesar Chavez Plaza (910 I St, City of Sacramento,
# "PARK, GREENBELT, ETC", score 35.1) and the Capitol grounds (1010 L St,
# State of California, "STATE USE-EXEMPT", score 22.3). Government-owned
# parcels also aren't vacancy-fee targets in the first place (fees apply to
# private owners), so excluding them isn't just a park-specific patch -- see
# hackathon_data/qc_park_exclusion.py for the analogous fix to the
# tier-classified (non-predicted) vacant_parcels set.
_PUBLIC_OWNER_PATTERN = re.compile(
    r"^CITY OF |^COUNTY OF |^STATE OF CALIFORNIA|^UNITED STATES OF AMERICA"
    r"|SCHOOL DISTRICT|COMMUNITY SERVICES DISTRICT|IRRIGATION DISTRICT"
    r"|WATER DISTRICT|FIRE DISTRICT|SEWER DISTRICT"
    r"|RECREATION.*PARK.*(DISTRICT|DIST)\b|PARK.*RECREATION.*(DISTRICT|DIST)\b"
)
_PUBLIC_USE_PATTERN = re.compile(r"PARK, GREENBELT|GREENBELT|USE-EXEMPT")


def is_public_land(owner, use_desc) -> bool:
    owner = owner.upper() if isinstance(owner, str) else ""
    use_desc = use_desc.upper() if isinstance(use_desc, str) else ""
    return bool(_PUBLIC_OWNER_PATTERN.search(owner)) or bool(_PUBLIC_USE_PATTERN.search(use_desc))


def attribute_with_parcel_id(gpd, calls, parcels):
    """Nearest-parcel join that KEEPS the parcel row id (so we can group by it)."""
    parcels_idx = parcels.reset_index(drop=True)
    parcels_idx["pid"] = np.arange(len(parcels_idx))
    joined = gpd.sjoin_nearest(
        calls[["CategoryName", "geometry"]],
        parcels_idx[["pid", "geometry"]],
        how="left", max_distance=S.SNAP_DISTANCE_FT, distance_col="dist_ft",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined[joined["pid"].notna()].copy()
    joined["pid"] = joined["pid"].astype(int)
    return joined, parcels_idx


def compute_lifts(joined, parcels_idx):
    """Per-category vacancy lift + the (pid -> set of categories) presence table."""
    is_vacant = parcels_idx["is_vacant"].to_numpy()
    base_rate = float(is_vacant.mean())

    # One row per (parcel, category) it received — de-duplicated.
    pc = joined[["pid", "CategoryName"]].drop_duplicates()
    pc = pc.join(parcels_idx["is_vacant"], on="pid")

    grp = pc.groupby("CategoryName")["is_vacant"]
    lift = pd.DataFrame({
        "support": grp.size(),
        "vacant_rate": grp.mean(),
    })
    lift = lift[lift["support"] >= MIN_SUPPORT_PARCELS]
    lift["lift"] = lift["vacant_rate"] / base_rate
    lift = lift.sort_values("lift", ascending=False)
    # weight = positive log2(lift) — "bits of vacancy evidence" per signal.
    weight = {c: max(np.log2(l), 0.0) for c, l in lift["lift"].items()}
    return lift, weight, base_rate, pc


def score_parcels(pc, weight, n_parcels):
    """Sum signal weights per parcel -> a transparent vacancy-likelihood score."""
    pc = pc.copy()
    pc["w"] = pc["CategoryName"].map(weight).fillna(0.0)
    score = pc.groupby("pid")["w"].sum()
    out = np.zeros(n_parcels, dtype=float)
    out[score.index.to_numpy()] = score.to_numpy()
    return out


def _trim(label, n=40):
    label = label.replace(" - ", " – ")
    return label if len(label) <= n else label[: n - 1].rstrip() + "…"


def fig_signals(lift, base_rate):
    """Horizontal bars: the 311 categories that most predict vacancy."""
    top = lift.head(10).iloc[::-1]
    fams = [S._family(c) for c in top.index]
    colors = [color for _l, color in fams]

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.subplots_adjust(left=0.34, top=0.84, bottom=0.12)
    bars = ax.barh(range(len(top)), top["lift"], color=colors, edgecolor="white",
                   height=0.72)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_trim(S._short(c)) for c in top.index], fontsize=9.5)
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1)
    for i, (l, n) in enumerate(zip(top["lift"], top["support"])):
        ax.text(l + top["lift"].max() * 0.012, i,
                f"{l:.1f}×   ({int(n):,} parcels)", va="center", fontsize=9,
                color="#15191c", fontweight="bold")
    ax.set_xlabel("Vacancy lift  (× more likely vacant than a random parcel · "
                  "dashed line = county-average rate)")
    ax.set_xlim(0, top["lift"].max() * 1.30)
    S._titled(ax, "Which 311 signals predict vacancy",
              f"A parcel with one of these call types is far more likely to be "
              f"vacant than the {base_rate*100:.1f}% county baseline", ha="left")
    # Legend explaining the bar colours (complaint families that appear).
    import matplotlib.patches as mpatches
    seen, handles = set(), []
    for _pre, label, color in S._FAMILIES:
        if label in {l for l, _c in fams} and label not in seen:
            seen.add(label)
            handles.append(mpatches.Patch(color=color, label=label))
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
                  title="Complaint family")
    ax.grid(axis="y", visible=False)
    S._footer(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "predicted_vacancy_signals.png")
    plt.close(fig)
    print("Saved predicted_vacancy_signals.png")


def main():
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("ERROR: geopandas is required. pip install geopandas")

    ok = (S._require(S.CALLS_GPKG, "Download SacCounty_SalesForce311_calls.gpkg into data/")
          and S._require(S.PARCELS_GPKG, "Download parcels_simplified.gpkg into hackathon_data/")
          and S._require(S.VACANT_CSV, "Download vacant_parcels.csv into hackathon_data/"))
    if not ok:
        sys.exit("\nMissing inputs — see messages above.")

    calls = S.load_hs_calls(gpd)
    parcels, _vac = S.load_parcels_with_vacancy(gpd)

    print("Attributing calls to parcels for the signal model...")
    joined, parcels_idx = attribute_with_parcel_id(gpd, calls, parcels)
    lift, weight, base_rate, pc = compute_lifts(joined, parcels_idx)
    print(f"  base vacancy rate {base_rate*100:.1f}%; "
          f"{len(lift)} predictive categories (support >= {MIN_SUPPORT_PARCELS})")
    print("  top signals:")
    for c, row in lift.head(6).iterrows():
        print(f"    {row['lift']:.1f}x  {c}  (n={int(row['support']):,})")

    scores = score_parcels(pc, weight, len(parcels_idx))
    parcels_idx["vac_score"] = scores

    # Threshold: a non-vacant parcel is a "candidate" if it scores at least as
    # high as the median known-vacant parcel that carries any signal.
    known_vac_scored = parcels_idx.loc[
        parcels_idx["is_vacant"] & (parcels_idx["vac_score"] > 0), "vac_score"]
    threshold = float(known_vac_scored.median()) if len(known_vac_scored) else 1.0

    is_public = parcels_idx.apply(
        lambda r: is_public_land(r.get("ASSESSEE_OWNER_NAME_1"), r.get("USE_CODE_MUNI_DESC")), axis=1
    )
    n_public_flagged = int(((~parcels_idx["is_vacant"]) & (parcels_idx["vac_score"] >= threshold) & is_public).sum())

    cand_mask = (~parcels_idx["is_vacant"]) & (parcels_idx["vac_score"] >= threshold) & (~is_public)
    candidates = parcels_idx[cand_mask].copy()
    print(f"  excluded {n_public_flagged:,} public/government-owned parcels "
          "that scored above threshold (parks, plazas, government buildings)")

    # Precision-like sanity check: among ALL parcels scoring >= threshold, what
    # share are already known vacant? (higher = the score tracks real vacancy).
    scored_hi = parcels_idx[(parcels_idx["vac_score"] >= threshold) & (~is_public)]
    precision_proxy = float(scored_hi["is_vacant"].mean()) if len(scored_hi) else 0.0

    print(f"  score threshold {threshold:.2f}; "
          f"{len(candidates):,} candidate (hidden) vacancies flagged")
    print(f"  precision proxy: {precision_proxy*100:.0f}% of high-score parcels "
          f"are already coded vacant")

    # Export candidates (EPSG:4326) for the pyQGIS map.
    PRED_GPKG.parent.mkdir(parents=True, exist_ok=True)
    out = candidates[["vac_score", "geometry"]].to_crs("EPSG:4326")
    out.to_file(PRED_GPKG, layer="candidates", driver="GPKG")
    print(f"  wrote {len(out):,} candidate parcels -> {PRED_GPKG}")

    fig_signals(lift, base_rate)

    # Append to findings.json (don't clobber the synthesis keys).
    findings = {}
    if FINDINGS.exists():
        findings = json.loads(FINDINGS.read_text())
    findings.update({
        "predict_base_vacancy_rate_pct": round(base_rate * 100, 1),
        "predict_n_predictive_categories": int(len(lift)),
        "predict_top_signal": lift.index[0],
        "predict_top_signal_lift": round(float(lift["lift"].iloc[0]), 1),
        "predict_n_candidate_vacancies": int(len(candidates)),
        "predict_high_score_precision_pct": round(precision_proxy * 100, 0),
    })
    FINDINGS.write_text(json.dumps(findings, indent=2))
    print("Updated findings.json")
    print(f"\nDone. The 311 signal flags ~{len(candidates):,} hidden candidate "
          f"vacancies beyond the {int(parcels_idx['is_vacant'].sum()):,} coded set.")


if __name__ == "__main__":
    main()
