"""
Vacancy x 311 Spatial Synthesis
===============================================================================
Joins Sacramento 311 service requests to parcels and asks the question that
matters for vacancy-fee policy: *do vacant parcels generate a disproportionate
share of health-&-safety / nuisance public-service demand?*

(Terminology note: research in urban land use avoids "blight," which carries
racist connotations tied to urban-renewal-era clearance. This project frames the
same 311 categories as "health & safety" / nuisance complaints, and the pattern
they reveal as vacant land being "high-maintenance".)

Unlike ``correlation_analysis.py`` (which measures how 311 categories co-occur
with one another at a shared address), this script performs the actual spatial
synthesis of two datasets:

    1. 311 calls   -> data/SacCounty_SalesForce311_calls.gpkg  (POINT geometry)
    2. parcels     -> hackathon_data/parcels_simplified.gpkg    (polygon geometry)
       vacant set  -> hackathon_data/vacant_parcels.csv         (APN list, 3 tiers)

Each health-&-safety 311 call is attributed to the nearest parcel (within a
configurable distance), parcels are flagged vacant/occupied, and a narrative
suite of figures is produced that walks from "how big is the problem" through
"vacant land is high-maintenance" to "the $70 flat fee does not cover the cost".

The script is defensive about column names and CRS, skips any figure whose
inputs are missing, and never assumes the heavy datasets are present until it
needs them. Run with no data and it tells you exactly what to download.

Output: PNG figures + a machine-readable findings.json in this directory.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# geopandas is only needed for the spatial steps; import lazily in main() so the
# module can be imported / linted without the geo stack installed.

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
HACK_DIR = PROJECT_ROOT / "hackathon_data"

CALLS_GPKG = DATA_DIR / "SacCounty_SalesForce311_calls.gpkg"
CALLS_LAYER = "SalesForce311"
PARCELS_GPKG = HACK_DIR / "parcels_simplified.gpkg"
# Prefer the QC-corrected export (public park/rec-district parcels removed,
# see hackathon_data/qc_vacancy_exclusions.py) when it's been generated locally.
VACANT_CSV = HACK_DIR / "vacant_parcels_qc.csv"
if not VACANT_CSV.exists():
    VACANT_CSV = HACK_DIR / "vacant_parcels.csv"
COUNCIL_SHP_DIR = DATA_DIR / "council_districts"

OUT_DIR = SCRIPT_DIR              # findings.json (machine-readable sidecar)
FIG_DIR = SCRIPT_DIR / "figures"  # all PNG figures land here

# ── Analysis parameters ──────────────────────────────────────────────────────

# Working CRS: California State Plane Zone 2, US feet. Distances come out in
# feet, which makes "within N feet of a parcel" intuitive.
WORK_CRS = "EPSG:2226"

# A 311 point is attributed to a parcel if it falls within this many feet of it.
# Many such calls (illegal dumping, abandoned vehicles, camps) sit in the
# right-of-way fronting a lot rather than inside the polygon, so we snap to the
# nearest parcel within a tolerance instead of requiring strict containment.
SNAP_DISTANCE_FT = 150

# Distance bands (feet) for the proximity / distance-decay figure.
DISTANCE_BANDS_FT = [0, 50, 150, 300, 600, 1200, 2640]  # last band ~ 1/2 mile

# Illustrative municipal cost to intake + respond to a single health-&-safety
# 311 case. Clearly labelled as an assumption in the figure; override on the CLI
# if you have a real per-case figure from the county.
COST_PER_CALL_USD = 125.0
FLAT_VACANCY_FEE_USD = 70.0

# ── Health & safety / nuisance category definition ───────────────────────────
# These are the 311 categories that plausibly signal neglect, abandonment, or
# the externalities of vacant land (code enforcement, encampments, illegal
# dumping, abandoned animals). Mirrors the tiers documented in ANALYSIS_NOTES.md.
# Used both to pre-filter the (1.5M-row) calls layer and to build the complaint-
# signature breakdown.

HS_LEVEL1 = ["Code Enforcement", "Homeless Camp", "Homeless Camp - Primary"]
HS_CATEGORYNAME = [
    "Solid Waste Illegal Dumping",
    "Solid Waste Code Enforcement Illegal Dumping",
    "Solid Waste Code Enforcement Receptacles",
    "Solid Waste Code Enforcement Receptacles - Residential",
    "Solid Waste Code Enforcement Receptacles - Commercial",
    "Animal Control Abandoned",
]

# The handful of categories that most directly name vacancy/abandonment. Used to
# highlight the "smoking gun" slice within the broader health-&-safety set.
DIRECT_VACANCY_CATS = [
    "Code Enforcement Housing - Boardup",
    "Code Enforcement Housing - Complaint",
    "Code Enforcement Emergency Housing Repair Program - Complaint",
    "Animal Control Abandoned",
]

# ── Visual identity ──────────────────────────────────────────────────────────

C_VACANT = "#E63946"      # red — vacant / high-maintenance
C_OCCUPIED = "#8D99AE"    # slate grey — occupied baseline
C_ACCENT = "#1D3557"      # deep navy — emphasis / text
C_GOLD = "#F4A259"        # amber — secondary series
C_GREEN = "#2A9D8F"       # teal — positive / fee line
GRID = "#E9ECEF"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CED4DA",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlecolor": C_ACCENT,
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "figure.dpi": 140,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

DOLLAR_FMT = FuncFormatter(lambda x, _: f"${x:,.0f}")
THOUSANDS_FMT = FuncFormatter(lambda x, _: f"{x:,.0f}")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("synthesis")


# ── Findings accumulator ─────────────────────────────────────────────────────

@dataclass
class Findings:
    """Collects the headline numbers so they can be reused in copy / socials."""
    values: dict = field(default_factory=dict)

    def add(self, key, value):
        self.values[key] = value
        return value

    def save(self, path):
        # numpy scalars are not JSON-serializable; coerce.
        def clean(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v
        with open(path, "w") as fh:
            json.dump({k: clean(v) for k, v in self.values.items()}, fh, indent=2)
        log.info("Saved findings.json")


# ── Loading / preparation ────────────────────────────────────────────────────

def _require(path: Path, hint: str):
    if not path.exists():
        log.error("MISSING: %s", path)
        log.error("  %s", hint)
        return False
    return True


def _sql_in(values):
    """Render a SQL IN (...) list with single-quote escaping."""
    escaped = ["'" + v.replace("'", "''") + "'" for v in values]
    return ", ".join(escaped)


def load_hs_calls(gpd):
    """Load only health-&-safety / nuisance 311 points, reprojected to WORK_CRS.

    Uses a pushed-down OGR ``where`` clause so we never materialize all 1.5M
    rows in memory.
    """
    where = (
        f"CategoryLevel1 IN ({_sql_in(HS_LEVEL1)}) "
        f"OR CategoryName IN ({_sql_in(HS_CATEGORYNAME)})"
    )
    log.info("Loading health & safety 311 calls (filtered at the source)...")
    t0 = time.time()
    cols = ["CategoryLevel1", "CategoryLevel2", "CategoryName",
            "DateCreated", "CouncilDistrictNumber", "ZIP", "Neighborhood"]
    try:
        calls = gpd.read_file(CALLS_GPKG, layer=CALLS_LAYER, where=where,
                              columns=cols)
    except Exception:
        # Older drivers may not honor `columns`; fall back to a full read.
        calls = gpd.read_file(CALLS_GPKG, layer=CALLS_LAYER, where=where)
    log.info("  %s calls in %.1fs", f"{len(calls):,}", time.time() - t0)

    if calls.crs is None:
        log.warning("  311 layer has no CRS; assuming EPSG:4326 (lon/lat).")
        calls = calls.set_crs("EPSG:4326")
    calls = calls.to_crs(WORK_CRS)
    calls = calls[~calls.geometry.is_empty & calls.geometry.notna()]

    # Parse the report date for the temporal figure. Format is unknown across
    # exports, so let pandas infer and silently drop unparseable rows later.
    if "DateCreated" in calls.columns:
        calls["report_date"] = pd.to_datetime(calls["DateCreated"], errors="coerce")
    return calls


def load_parcels_with_vacancy(gpd):
    """Load all parcel polygons and flag each as vacant or occupied.

    Vacancy is defined by membership in the curated 3-tier vacant set (matched
    on APN), which is broader than the GeoPackage's ``is_vacant_coded`` flag
    (Tier 1 only).
    """
    log.info("Loading parcels + vacancy flag...")
    t0 = time.time()
    parcels = gpd.read_file(PARCELS_GPKG)
    if parcels.crs is None:
        log.warning("  parcels layer has no CRS; assuming EPSG:2226.")
        parcels = parcels.set_crs(WORK_CRS)
    parcels = parcels.to_crs(WORK_CRS)

    apn_col = next((c for c in ("APN", "PARCEL_APN", "apn") if c in parcels.columns), None)
    if apn_col is None:
        sys.exit("ERROR: no APN column found in parcels layer.")
    parcels["apn_key"] = _normalize_apn(parcels[apn_col])

    vac = pd.read_csv(VACANT_CSV, usecols=lambda c: c in
                      ("PARCEL_APN", "vacancy_tier", "VAL_ASSD_LAND"),
                      low_memory=False)
    vac["apn_key"] = _normalize_apn(vac["PARCEL_APN"])
    vac_keys = set(vac["apn_key"])
    tier_by_apn = dict(zip(vac["apn_key"], vac.get("vacancy_tier", pd.Series(dtype=str))))

    parcels["is_vacant"] = parcels["apn_key"].isin(vac_keys)
    parcels["vacancy_tier"] = parcels["apn_key"].map(tier_by_apn)
    log.info("  %s parcels (%s vacant) in %.1fs",
             f"{len(parcels):,}", f"{int(parcels['is_vacant'].sum()):,}",
             time.time() - t0)
    return parcels, vac


def _normalize_apn(series):
    """Coerce APN values to a comparable zero-padded string key.

    The GeoPackage stores APNs as 14-digit zero-padded strings
    (e.g. ``00100110010000``); the CSV often stores them as int64, which drops
    any leading zeros (``100110010000``). Floats may also sneak in. Strip any
    trailing ``.0`` and whitespace, then left-pad back to 14 digits so both
    representations compare equal. Without the pad, every vacant parcel whose
    APN begins with a zero silently fails to match (~45% of the vacant set).
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.zfill(14)
    return s


# ── Spatial attribution ──────────────────────────────────────────────────────

def attribute_calls_to_parcels(gpd, calls, parcels):
    """Snap each call to its nearest parcel within SNAP_DISTANCE_FT.

    Returns ``calls`` with added columns: ``parcel_idx``, ``is_vacant``,
    ``vacancy_tier``, ``dist_ft`` (distance to the matched parcel).
    """
    log.info("Attributing calls to nearest parcel (<= %d ft)...", SNAP_DISTANCE_FT)
    t0 = time.time()
    parcels_idx = parcels.reset_index(drop=True)
    joined = gpd.sjoin_nearest(
        calls, parcels_idx[["is_vacant", "vacancy_tier", "geometry"]],
        how="left", max_distance=SNAP_DISTANCE_FT, distance_col="dist_ft",
    )
    # sjoin_nearest can emit duplicate rows on exact ties; keep the first.
    joined = joined[~joined.index.duplicated(keep="first")]
    matched = joined["is_vacant"].notna()
    log.info("  %s of %s calls matched a parcel in %.1fs",
             f"{int(matched.sum()):,}", f"{len(calls):,}", time.time() - t0)
    return joined


def distance_to_nearest_vacant(gpd, calls, parcels):
    """Distance (ft) from every health-&-safety call to the nearest *vacant* parcel."""
    vacant = parcels[parcels["is_vacant"]]
    if vacant.empty:
        return None
    nearest = gpd.sjoin_nearest(
        calls[["geometry"]], vacant[["geometry"]], how="left",
        distance_col="dist_to_vacant_ft",
    )
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    return nearest["dist_to_vacant_ft"].to_numpy()


# ── Figure helpers ───────────────────────────────────────────────────────────

def _footer(fig, text="Source: Sacramento County 311 (SalesForce311) x parcel assessor data  ·  vacancyfee.org"):
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=8, color="#888888")


def _titled(ax, title, subtitle=None, ha="center"):
    """Stack a bold title and a muted subtitle above the axes without overlap."""
    ax.set_title(title, pad=38 if subtitle else 14)
    if subtitle:
        x = 0.5 if ha == "center" else 0.0
        ax.text(x, 1.012, subtitle, transform=ax.transAxes, ha=ha, va="bottom",
                fontsize=10.5, color="#666666")


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved %s", name)


def _bar_labels(ax, bars, fmt="{:.0f}", dy=0.0, fontsize=10, color=C_ACCENT):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
                color=color)


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_burden_per_parcel(parcels, joined, F):
    """HEADLINE: health-&-safety 311 calls per parcel, vacant vs occupied."""
    n_vacant = int(parcels["is_vacant"].sum())
    n_occupied = int((~parcels["is_vacant"]).sum())

    matched = joined[joined["is_vacant"].notna()]
    calls_vacant = int((matched["is_vacant"] == True).sum())  # noqa: E712
    calls_occupied = int((matched["is_vacant"] == False).sum())  # noqa: E712

    rate_vacant = calls_vacant / n_vacant if n_vacant else 0
    rate_occupied = calls_occupied / n_occupied if n_occupied else 0
    multiplier = rate_vacant / rate_occupied if rate_occupied else float("nan")

    F.add("n_vacant_parcels", n_vacant)
    F.add("n_occupied_parcels", n_occupied)
    F.add("hs_calls_on_vacant", calls_vacant)
    F.add("hs_calls_on_occupied", calls_occupied)
    F.add("calls_per_vacant_parcel", round(rate_vacant, 3))
    F.add("calls_per_occupied_parcel", round(rate_occupied, 3))
    F.add("vacant_hs_multiplier", round(multiplier, 2))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(["Occupied\nparcels", "Vacant\nparcels"],
                  [rate_occupied, rate_vacant],
                  color=[C_OCCUPIED, C_VACANT], width=0.6, edgecolor="white")
    _bar_labels(ax, bars, fmt="{:.2f}", dy=rate_vacant * 0.01)
    ax.set_ylabel("Health & safety 311 calls per parcel")
    sub = (f"A vacant parcel draws {multiplier:.1f}x more health & safety 311 "
           f"calls than an occupied one")
    _titled(ax, "Vacant parcels are high-maintenance", sub)
    ax.margins(y=0.18)
    _footer(fig, "Health & safety 311 calls = code-enforcement, illegal-dumping, "
                 "encampment & abandoned-animal reports\n"
                 "Source: Sacramento County 311 (SalesForce311) x parcel assessor "
                 "data  ·  vacancyfee.org")
    _save(fig, "fig1_burden_per_parcel.png")


def fig_share_mismatch(parcels, joined, F):
    """Vacant land is a small share of parcels but a large share of H&S calls."""
    n_total = len(parcels)
    n_vacant = int(parcels["is_vacant"].sum())
    matched = joined[joined["is_vacant"].notna()]
    calls_total = len(matched)
    calls_vacant = int((matched["is_vacant"] == True).sum())  # noqa: E712

    share_parcels = 100 * n_vacant / n_total if n_total else 0
    share_calls = 100 * calls_vacant / calls_total if calls_total else 0
    F.add("vacant_share_of_parcels_pct", round(share_parcels, 1))
    F.add("vacant_share_of_hs_calls_pct", round(share_calls, 1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    metrics = ["Share of all\nparcels", "Share of all health &\nsafety 311 calls"]
    vac_vals = [share_parcels, share_calls]
    other_vals = [100 - share_parcels, 100 - share_calls]
    y = np.arange(len(metrics))
    ax.barh(y, vac_vals, color=C_VACANT, label="Vacant land", edgecolor="white")
    ax.barh(y, other_vals, left=vac_vals, color=C_OCCUPIED,
            label="Everything else", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.invert_yaxis()
    for yi, v in zip(y, vac_vals):
        ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", ha="left",
                fontweight="bold", color=C_VACANT)
    _titled(ax, "A disproportionate burden",
            f"Vacant parcels are {share_parcels:.1f}% of the county's land base "
            f"but draw {share_calls:.1f}% of its health & safety complaints", ha="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    _footer(fig)
    _save(fig, "fig2_share_mismatch.png")


def fig_call_signature(joined, F):
    """What kinds of H&S calls land on vacant parcels (the 'signature')."""
    matched = joined[joined["is_vacant"] == True]  # noqa: E712
    if matched.empty or "CategoryName" not in matched.columns:
        log.info("  skipping call signature (no matched vacant calls).")
        return
    top = matched["CategoryName"].value_counts().head(15).sort_values()
    F.add("top_vacant_call_category", top.index[-1])
    F.add("top_vacant_call_category_count", int(top.iloc[-1]))

    fig, ax = plt.subplots(figsize=(11, 8))
    fam = [_family(c) for c in top.index]
    colors = [color for _label, color in fam]
    bars = ax.barh(range(len(top)), top.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_short(c) for c in top.index], fontsize=9)
    ax.xaxis.set_major_formatter(THOUSANDS_FMT)
    ax.set_xlabel("311 calls on vacant parcels")
    _titled(ax, "The complaint signature of vacant land",
            "What residents actually report on vacant parcels, "
            "by complaint type", ha="left")
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f"{int(v):,}", va="center", fontsize=8,
                color="#555555")
    # Build a legend from the families that actually appear, in _FAMILIES order.
    import matplotlib.patches as mpatches
    seen, handles = set(), []
    for _pre, label, color in _FAMILIES:
        if label in {l for l, _c in fam} and label not in seen:
            seen.add(label)
            handles.append(mpatches.Patch(color=color, label=label))
    if any(l == "Other" for l, _c in fam):
        handles.append(mpatches.Patch(color=C_OCCUPIED, label="Other"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              title="Complaint family")
    ax.grid(axis="y", visible=False)
    _footer(fig)
    _save(fig, "fig3_call_signature.png")


def fig_distance_decay(dist_to_vacant, F):
    """Health-&-safety calls concentrate near vacant parcels (distance-decay)."""
    if dist_to_vacant is None:
        log.info("  skipping distance-decay (no vacant parcels).")
        return
    d = dist_to_vacant[~np.isnan(dist_to_vacant)]
    if len(d) == 0:
        return
    bands = DISTANCE_BANDS_FT
    labels = [f"{bands[i]}–{bands[i+1]} ft" for i in range(len(bands) - 1)]
    labels.append(f"{bands[-1]}+ ft")
    edges = bands + [np.inf]
    counts, _ = np.histogram(d, bins=edges)
    pct = 100 * counts / counts.sum()
    within_300 = float(100 * (d <= 300).mean())
    F.add("pct_hs_calls_within_300ft_of_vacant", round(within_300, 1))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(range(len(pct)), pct, color=C_VACANT, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(labels)))
    # Horizontal labels (they're short) so they don't collide with the axis
    # label or the footer; a touch smaller to fit seven bands.
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylabel("Share of health & safety 311 calls")
    ax.set_xlabel("Distance to nearest vacant parcel", labelpad=10)
    _bar_labels(ax, bars, fmt="{:.0f}%", dy=pct.max() * 0.01, fontsize=9)
    _titled(ax, "Health & safety calls cluster at the edge of vacant land",
            f"{within_300:.0f}% of all health & safety 311 calls fall within "
            f"300 ft of a vacant parcel", ha="left")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.18)
    _footer(fig)
    _save(fig, "fig4_distance_decay.png")


def fig_council_synthesis(gpd, parcels, joined, F):
    """Per council district: vacant-parcel count vs health-&-safety-call volume."""
    council = _load_council(gpd)
    if council is None:
        log.info("  skipping council-district figures (boundary file absent).")
        return

    # Assign vacant parcels and matched calls to districts via spatial join so
    # both series share one consistent geography.
    dist_col = _council_label_col(council)
    vac = parcels[parcels["is_vacant"]][["geometry"]].copy()
    vac = vac.set_geometry("geometry")
    vac_pts = vac.copy()
    vac_pts["geometry"] = vac_pts.geometry.representative_point()
    vac_j = gpd.sjoin(vac_pts, council[[dist_col, "geometry"]], how="inner", predicate="within")
    vac_by_dist = vac_j.groupby(dist_col).size()

    matched = joined[joined["is_vacant"].notna()][["geometry"]].copy()
    call_j = gpd.sjoin(matched, council[[dist_col, "geometry"]], how="inner", predicate="within")
    calls_by_dist = call_j.groupby(dist_col).size()

    df = pd.DataFrame({"vacant": vac_by_dist, "calls": calls_by_dist}).fillna(0)
    df = df[(df["vacant"] > 0) | (df["calls"] > 0)].sort_values("calls", ascending=False)
    if df.empty:
        log.info("  skipping council figures (no overlap).")
        return

    # --- Paired bars ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.4
    b1 = ax1.bar(x - w / 2, df["vacant"], w, color=C_OCCUPIED, label="Vacant parcels")
    ax1.set_ylabel("Vacant parcels", color=C_OCCUPIED)
    ax1.tick_params(axis="y", labelcolor=C_OCCUPIED)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, df["calls"], w, color=C_VACANT, label="Health & safety 311 calls")
    ax2.set_ylabel("Health & safety 311 calls", color=C_VACANT)
    ax2.tick_params(axis="y", labelcolor=C_VACANT)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in df.index], rotation=45, ha="right")
    ax1.set_xlabel("Council district")
    ax1.set_title("Where vacancy and health & safety calls overlap", pad=26)
    # Dual-axis caveat: the two y-axes are scaled independently, so don't compare
    # bar heights directly — compare the pattern across districts.
    ax1.text(0.0, 1.02,
             "Grey = vacant parcels (left axis)  ·  Red = health & safety 311 "
             "calls (right axis)  ·  axes scaled independently",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=9.5,
             color="#666666")
    ax1.grid(axis="x", visible=False)
    # Legend inside the plot (upper-right has headroom above the shorter bars) so
    # it no longer collides with the title.
    ax1.legend([b1, b2], ["Vacant parcels", "Health & safety 311 calls"],
               loc="upper right", frameon=True, facecolor="white",
               framealpha=0.92, edgecolor="#cccccc")
    _footer(fig)
    _save(fig, "fig5_council_paired.png")

    # --- Scatter + regression ---
    if len(df) >= 3:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(df["vacant"], df["calls"], s=90, color=C_VACANT,
                   edgecolor="white", zorder=3)
        for d, r in df.iterrows():
            ax.annotate(str(d), (r["vacant"], r["calls"]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points", color="#555555")
        m, b = np.polyfit(df["vacant"], df["calls"], 1)
        xs = np.linspace(df["vacant"].min(), df["vacant"].max(), 50)
        ax.plot(xs, m * xs + b, color=C_ACCENT, linestyle="--", linewidth=1.5,
                label="best-fit trend")
        r = float(np.corrcoef(df["vacant"], df["calls"])[0, 1])
        F.add("council_vacant_vs_calls_r", round(r, 3))
        ax.set_xlabel("Vacant parcels in district")
        ax.set_ylabel("Health & safety 311 calls in district")
        _titled(ax, "More vacancy, more health & safety complaints",
                "Each dot is a council district — districts with more vacant land "
                "field more 311 calls", ha="left")
        ax.text(0.05, 0.93, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=C_ACCENT)
        ax.text(0.05, 0.875, "correlation · 1.0 = perfect", transform=ax.transAxes,
                fontsize=9, color="#777777")
        ax.legend(loc="lower right", frameon=False, fontsize=9.5)
        _footer(fig)
        _save(fig, "fig6_council_scatter.png")


def fig_temporal_trend(joined, F):
    """Monthly health-&-safety-call volume on vacant parcels over time.

    (The old hexbin hot-spot map that used to sit here was removed — the pyQGIS
    map suite in ../maps/ renders legible, basemapped hot-spot maps instead.)
    """
    if "report_date" not in joined.columns:
        return
    matched = joined[(joined["is_vacant"] == True) & joined["report_date"].notna()]  # noqa: E712
    if len(matched) < 24:
        log.info("  skipping temporal trend (insufficient dated calls).")
        return
    monthly = matched.set_index("report_date").resample("MS").size()
    # Trim sparse leading/trailing months that distort the line.
    monthly = monthly[monthly.cumsum() > 0]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(monthly.index, monthly.values, color=C_VACANT, alpha=0.18)
    ax.plot(monthly.index, monthly.values, color=C_VACANT, linewidth=2,
            label="Monthly calls")
    # 12-month rolling average for the trend read.
    roll = monthly.rolling(12, min_periods=3).mean()
    ax.plot(roll.index, roll.values, color=C_ACCENT, linewidth=2, linestyle="--",
            label="12-month average")
    ax.set_ylabel("Health & safety 311 calls on vacant parcels / month")
    ax.set_title("The problem isn't going away")

    # HONESTY CAVEAT: the SalesForce311 export is sparse before 2023 (the county
    # migrated reporting systems then), so the near-zero pre-2023 values reflect
    # data coverage, not lower demand. Shade and label that window so the ramp is
    # never mistaken for real growth.
    cutoff = pd.Timestamp("2023-01-01")
    # DateCreated carries a UTC 'Z', so the resampled index is tz-aware; match it
    # to avoid a tz-naive/aware comparison error.
    if getattr(monthly.index, "tz", None) is not None:
        cutoff = cutoff.tz_localize(monthly.index.tz)
    if monthly.index.min() < cutoff:
        ax.axvspan(monthly.index.min(), cutoff, color="#CED4DA", alpha=0.30,
                   zorder=0)
        ax.axvline(cutoff, color="#888888", linestyle=":", linewidth=1.2)
        ymax = max(monthly.max(), 1)
        ax.text(monthly.index.min(), ymax * 0.96,
                "  Partial 311 coverage before 2023\n"
                "  (reporting-system migration) —\n"
                "  not lower demand",
                fontsize=8.5, color="#777777", va="top", ha="left", style="italic")

    # Place the legend in the empty pre-2023 area BELOW the coverage caveat so
    # the two don't overlap (the curve is near zero there).
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.72))
    ax.margins(x=0.01)
    F.add("temporal_months_covered", int(len(monthly)))
    _footer(fig)
    _save(fig, "fig7_temporal_trend.png")


def fig_cost_vs_fee(F):
    """The annual public burden of the H&S calls that vacant land draws.

    Reframed (deliberately) away from a flat-fee-vs-cost comparison: at any
    defensible per-call cost the $70 fee actually *exceeds* the narrow 311
    response cost, so that framing undercut the argument. The honest, stronger
    story is the *scale of the recurring externality* — what the public spends
    every year answering health & safety calls on vacant land, and how much of
    that is the excess attributable to vacancy (the 3x multiplier in dollars).
    """
    calls_vacant = F.values.get("hs_calls_on_vacant")
    years = F.values.get("data_years") or 1.0
    if not calls_vacant:
        return
    annual_calls = calls_vacant / years
    total_cost = annual_calls * COST_PER_CALL_USD

    # Excess attributable to vacancy: calls above the occupied-parcel baseline
    # rate, i.e. the share that wouldn't exist if vacant land behaved like the
    # rest of the parcel base. This is the "cost of the vacancy externality."
    rate_v = F.values.get("calls_per_vacant_parcel", 0) / years
    rate_o = F.values.get("calls_per_occupied_parcel", 0) / years
    n_vacant = F.values.get("n_vacant_parcels", 0)
    excess_cost = max(rate_v - rate_o, 0) * n_vacant * COST_PER_CALL_USD

    F.add("annual_hs_calls_on_vacant", round(annual_calls))
    F.add("annual_public_cost_vacant_hs_usd", round(total_cost))
    F.add("annual_excess_cost_attributable_to_vacancy_usd", round(excess_cost))

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Every health & safety 311\ncall on vacant parcels",
              "The excess vs. if vacant land\nbehaved like occupied land"]
    bars = ax.bar(labels, [total_cost, excess_cost],
                  color=[C_VACANT, C_ACCENT], width=0.6, edgecolor="white")
    _bar_labels(ax, bars, fmt="${:,.0f}", dy=total_cost * 0.01)
    ax.yaxis.set_major_formatter(DOLLAR_FMT)
    ax.set_ylabel("Public cost per year (county-wide)")
    _titled(ax, "What vacant land costs the public every year",
            f"Illustrative at ${COST_PER_CALL_USD:,.0f}/call · "
            f"~{annual_calls:,.0f} health & safety 311 calls a year on vacant "
            f"parcels · before cleanup, abatement, or lost tax base")
    ax.margins(y=0.18)
    _footer(fig, "The navy bar is the share of the red total attributable to "
                 "vacancy (calls above the occupied-parcel rate)\n"
                 "Source: Sacramento County 311 (SalesForce311) x parcel assessor "
                 "data  ·  vacancyfee.org")
    _save(fig, "fig8_cost_vs_fee.png")


# ── Small utilities ──────────────────────────────────────────────────────────

# Family grouping for the complaint-signature chart: a readable label + a color
# per top-level complaint family. Used to colour the bars AND build a real
# legend, so the chart no longer references a "red" highlight that never appears.
# Priority and regular encampment calls share a colour, so they share ONE legend
# label too (two identical red swatches read as a mistake otherwise).
_FAMILIES = [
    ("Homeless Camp - Primary", "Homeless camp / encampment", C_VACANT),
    ("Homeless Camp", "Homeless camp / encampment", C_VACANT),
    ("Code Enforcement", "Code enforcement", C_ACCENT),
    ("Solid Waste", "Illegal dumping / waste", C_GOLD),
    ("Animal Control", "Abandoned animals", C_GREEN),
]


def _family(cat: str):
    """Return (family label, color) for a CategoryName."""
    for pre, label, color in _FAMILIES:
        if cat.startswith(pre):
            return label, color
    return "Other", C_OCCUPIED


def _short(cat: str) -> str:
    """Human-readable label for a 311 CategoryName (no cryptic HC/CE/SW codes)."""
    fams = {
        "Homeless Camp - Primary": "Homeless camp",
        "Homeless Camp": "Homeless camp",
        "Code Enforcement": "Code enf.",
        "Solid Waste": "Dumping",
        "Animal Control": "Animal",
    }
    for pre, short in fams.items():
        if cat.startswith(pre):
            rest = cat[len(pre):].strip()
            rest = (rest.replace("HC-Trash SPD", "trash")
                        .replace("HC-Trash", "trash")
                        .replace("HC-Primary", "priority report")
                        .replace("Homeless Encampment Blocking Sidewalk",
                                 "encampment on sidewalk")
                        .strip())
            return f"{short} — {rest}" if rest else short
    return cat


def _load_council(gpd):
    if not COUNCIL_SHP_DIR.exists():
        return None
    shps = list(COUNCIL_SHP_DIR.glob("*.shp"))
    if not shps:
        return None
    council = gpd.read_file(shps[0])
    if council.crs is None:
        council = council.set_crs("EPSG:4326")
    return council.to_crs(WORK_CRS)


def _council_label_col(council):
    for c in ("DISTRICT", "District", "DISTNUM", "CouncilDistrictNumber",
              "DISTRICT_N", "NAME", "Name"):
        if c in council.columns:
            return c
    # Fall back to the first non-geometry column.
    return [c for c in council.columns if c != "geometry"][0]


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    global COST_PER_CALL_USD, SNAP_DISTANCE_FT
    import argparse
    p = argparse.ArgumentParser(description="Vacancy x 311 spatial synthesis")
    p.add_argument("--cost-per-call", type=float, default=COST_PER_CALL_USD)
    p.add_argument("--snap-ft", type=int, default=SNAP_DISTANCE_FT)
    args = p.parse_args(argv)

    COST_PER_CALL_USD = args.cost_per_call
    SNAP_DISTANCE_FT = args.snap_ft

    ok = True
    ok &= _require(CALLS_GPKG, "Download SacCounty_SalesForce311_calls.gpkg into data/")
    ok &= _require(PARCELS_GPKG, "Download parcels_simplified.gpkg into hackathon_data/ (see DATA_DOWNLOAD.md)")
    ok &= _require(VACANT_CSV, "Download vacant_parcels.csv into hackathon_data/")
    if not ok:
        sys.exit("\nMissing inputs — see messages above. Files live on the project Google Drive.")

    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("ERROR: geopandas is required. pip install geopandas")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    F = Findings()

    calls = load_hs_calls(gpd)
    parcels, _vac = load_parcels_with_vacancy(gpd)

    # Time window covered by the calls — used to annualize per-parcel rates.
    if "report_date" in calls.columns and calls["report_date"].notna().any():
        span_days = (calls["report_date"].max() - calls["report_date"].min()).days
        F.add("data_years", round(max(span_days / 365.25, 1.0), 2))

    joined = attribute_calls_to_parcels(gpd, calls, parcels)
    dist_to_vacant = distance_to_nearest_vacant(gpd, calls, parcels)

    # Narrative arc -----------------------------------------------------------
    fig_burden_per_parcel(parcels, joined, F)      # 1. the hook
    fig_share_mismatch(parcels, joined, F)         # 2. disproportionate burden
    fig_call_signature(joined, F)                  # 3. what the calls look like
    fig_distance_decay(dist_to_vacant, F)          # 4. proximity proof
    fig_council_synthesis(gpd, parcels, joined, F) # 5/6. geography
    fig_temporal_trend(joined, F)                  # 7. it persists
    fig_cost_vs_fee(F)                             # 8. the policy punchline
    # (the old hexbin hot-spot map is gone — see the pyQGIS maps in ../maps/)

    F.save(OUT_DIR / "findings.json")
    log.info("\nDone. Figures -> %s/figures/, findings.json -> %s/",
             OUT_DIR.name, OUT_DIR.name)
    log.info("Headline: vacant parcels draw %.1fx the health & safety calls of occupied land.",
             F.values.get("vacant_hs_multiplier", float("nan")))


if __name__ == "__main__":
    main()
