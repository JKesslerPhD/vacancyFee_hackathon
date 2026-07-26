"""
Citywide vacancy data + charts, with a District 3 (Karina Talamantes)
highlight -- feeds district_briefs/build_district3_brief.py.

Pulls citywide totals from district_revenue_summary.csv, District 3's own
slice of parcel_revenue_estimates.csv, joins SITE_ADDR from
hackathon_data/parcels_trimmed.csv for a District 3 highlight table, and
writes:

    revenue_impact/results/district3_vacant_properties.csv
        Every District 3 vacant/underused parcel this analysis flags, with
        address, zoning, market value estimate, years since last sale.

    revenue_impact/results/district3_summary.json
        Citywide totals + District 3's own numbers, for the brief.

    revenue_impact/figures/district3_*.png
        Print-resolution (300 DPI) charts in the Vacancy Fee Project print
        palette (brick red / black / cream / teal -- see PRINT_* colors
        below, distinct from the interactive map's web palette).

Run after revenue_impact/estimate_lost_revenue.py.

Data note: every parcel in parcel_revenue_estimates.csv has BUILDING_SQFT
== 0 (checked against hackathon_data/parcels_trimmed.csv) -- the QC pass in
hackathon_data/qc_vacancy_exclusions.py drops any Tier 2/3 parcel whose use
code implies a structure, on the assumption that a zero-improvement-value
parcel with a "real building" use code is more likely a data quality
artifact than a genuinely vacant building. That's a defensible call, but it
also means this dataset currently has zero examples of boarded-up vacant
commercial buildings, only vacant land -- something to flag if the goal is
to point at storefronts, not lots. The 3,808 parcels that QC pass excluded
as "use_code_implies_structure_not_vacant_land" (see
hackathon_data/qc_exclusion_report.csv) are where genuine boarded
buildings would be hiding, if any are in there.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

PARCEL_REVENUE_CSV = RESULTS_DIR / "parcel_revenue_estimates.csv"
DISTRICT_SUMMARY_CSV = RESULTS_DIR / "district_revenue_summary.csv"
PARCELS_TRIMMED_CSV = REPO_ROOT / "hackathon_data" / "parcels_trimmed.csv"

DISTRICT_LABEL = "District 3 (Karina Talamantes)"
DISTRICT_SHORT = {
    "District 6 (Eric Guerra)": "D6",
    "District 2 (Roger Dickinson)": "D2",
    "District 4 (Phil Pluckebaum)": "D4",
    "District 1 (Lisa Kaplan)": "D1",
    "District 5 (Caity Maple)": "D5",
    "District 3 (Karina Talamantes)": "D3",
    "District 7 (Rick Jennings II)": "D7",
    "District 8 (Mai Vang)": "D8",
}

# Vacancy Fee Project print brand -- sampled from the existing "NO MORE
# WASTED SPACE" one-pagers and the project logo. Distinct from the
# interactive map's web palette (--accent: #eb6834) by design.
PRINT_RED = "#A6352C"
PRINT_RED_DARK = "#7A2620"
PRINT_BLACK = "#1A1A1A"
PRINT_CREAM = "#F3EAD9"
PRINT_TEAL = "#2E5E58"
PRINT_GRAY = "#5A6068"
PRINT_GRAY_LIGHT = "#B8B2A8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": PRINT_GRAY,
    "axes.labelcolor": PRINT_BLACK,
    "text.color": PRINT_BLACK,
    "xtick.color": PRINT_BLACK,
    "ytick.color": PRINT_BLACK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def fmt_dollars(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.0f}K"
    return f"${v:,.0f}"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (all parcels citywide, District 3 parcels with address join, district summary table)."""
    all_parcels = pd.read_csv(PARCEL_REVENUE_CSV)
    district_summary = pd.read_csv(DISTRICT_SUMMARY_CSV, index_col=0)

    d3 = all_parcels[all_parcels["council_district"] == DISTRICT_LABEL].copy()
    print(f"District 3: {len(d3):,} vacant/underused parcels (citywide: {len(all_parcels):,})")

    addr = pd.read_csv(
        PARCELS_TRIMMED_CSV,
        usecols=["PARCEL_APN", "SITE_ADDR", "SITE_ZIP"],
        dtype={"PARCEL_APN": str},
        low_memory=False,
    )
    addr["PARCEL_APN"] = addr["PARCEL_APN"].str.zfill(14)
    addr = addr.drop_duplicates(subset="PARCEL_APN")
    d3["PARCEL_APN"] = d3["PARCEL_APN"].astype(str).str.zfill(14)
    d3 = d3.merge(addr, on="PARCEL_APN", how="left", suffixes=("", "_addr"))
    matched = d3["SITE_ADDR"].notna().sum()
    print(f"  {matched:,}/{len(d3):,} matched a site address ({matched / len(d3):.1%})")
    return all_parcels, d3, district_summary


def build_summary(all_parcels: pd.DataFrame, d3: pd.DataFrame, district_summary: pd.DataFrame) -> dict:
    city_row = district_summary.loc["CITYWIDE TOTAL"]
    d3_row = district_summary.loc[DISTRICT_LABEL]

    yrs_city = all_parcels["years_since_sale"].dropna()
    type_counts_city = all_parcels["property_type_group"].value_counts()

    summary = {
        "district": "District 3",
        "council_member": "Karina Talamantes",

        # Citywide (headline numbers for the brief)
        "city_n_vacant_properties": int(city_row["vacant_parcels"]),
        "city_n_commercial": int(type_counts_city.get("commercial", 0)),
        "city_n_industrial": int(type_counts_city.get("industrial", 0)),
        "city_n_vacant_land_residential": int(type_counts_city.get("vacant_land_residential", 0)),
        "city_total_property_tax_uplift": float(city_row["total_potential_property_tax_uplift"]),
        "city_total_sales_tax": float(city_row["total_estimated_annual_sales_tax_total"]),
        "city_total_sales_tax_city_share": float(city_row["total_estimated_annual_sales_tax_city"]),
        "city_n_with_sale_date": int(len(yrs_city)),
        "city_median_years_since_sale": float(yrs_city.median()) if len(yrs_city) else None,
        "city_pct_vacant_10plus_years": float((yrs_city >= 10).mean()) if len(yrs_city) else None,
        "city_pct_vacant_5plus_years": float((yrs_city >= 5).mean()) if len(yrs_city) else None,

        # District 3
        "d3_n_vacant_properties": int(d3_row["vacant_parcels"]),
        "d3_total_property_tax_uplift": float(d3_row["total_potential_property_tax_uplift"]),
        "d3_total_sales_tax": float(d3_row["total_estimated_annual_sales_tax_total"]),
        "d3_total_sales_tax_city_share": float(d3_row["total_estimated_annual_sales_tax_city"]),
    }
    return summary


def write_csv(d3: pd.DataFrame) -> None:
    cols = [
        "PARCEL_APN", "SITE_ADDR", "SITE_ZIP", "USE_CODE_STD_DESC_LPS", "ZONING",
        "property_type_group", "vacancy_tier", "LOT_SIZE_AREA",
        "VAL_ASSD", "est_market_value", "prop13_gap",
        "potential_property_tax_uplift", "commercial_eligible",
        "estimated_annual_sales_tax_total", "estimated_annual_sales_tax_city",
        "LAST_SALE_DATE_TRANSFER", "years_since_sale", "market_assessed_ratio",
    ]
    out = d3[cols].sort_values("potential_property_tax_uplift", ascending=False)
    target = RESULTS_DIR / "district3_vacant_properties.csv"
    out.to_csv(target, index=False)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({len(out):,} rows)")


def write_summary_json(summary: dict) -> None:
    target = RESULTS_DIR / "district3_summary.json"
    target.write_text(json.dumps(summary, indent=2))
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def chart_sales_tax_by_district(district_summary: pd.DataFrame) -> None:
    """Lost sales tax revenue, all 8 council districts, District 3 called
    out in red against the rest in black -- one chart giving both the
    citywide picture and District 3's place in it."""
    rows = district_summary.drop("CITYWIDE TOTAL").sort_values(
        "total_estimated_annual_sales_tax_total", ascending=True
    )
    labels = [DISTRICT_SHORT.get(idx, idx) for idx in rows.index]
    values = rows["total_estimated_annual_sales_tax_total"].values
    colors = [PRINT_RED if lbl == "D3" else PRINT_BLACK for lbl in labels]

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, "  " + fmt_dollars(v),
                 ha="left", va="center", fontsize=9, fontweight="bold", color=PRINT_BLACK)
    ax.set_xlabel("Lost sales tax revenue, per year", fontsize=9.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_dollars(v)))
    ax.set_xlim(0, values.max() * 1.35)
    ax.set_title("Every Council District Has This Problem", fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    target = FIGURES_DIR / "district3_revenue_potential.png"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def chart_vacancy_duration_citywide(all_parcels: pd.DataFrame, summary: dict) -> None:
    """Distribution of years-since-last-sale, citywide, computed only over
    the 6,518 parcels this analysis already flagged as vacant or
    underused -- never the city's full parcel roll."""
    yrs = all_parcels["years_since_sale"].dropna()
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=300)
    bins = [0, 2, 5, 10, 15, 20, 25, 35]
    labels = ["<2", "2-5", "5-10", "10-15", "15-20", "20-25", "25+"]
    counts, _ = np.histogram(yrs, bins=bins)
    bar_colors = [PRINT_TEAL if b < 10 else PRINT_RED for b in bins[:-1]]
    ax.bar(labels, counts, color=bar_colors, width=0.65)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Years since last recorded sale", fontsize=9.5)
    ax.set_ylabel("Vacant properties", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.set_title(
        "Citywide, Among Vacant Properties Only",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    target = FIGURES_DIR / "district3_vacancy_duration.png"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def chart_property_type_breakdown(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=300)
    # "Vacant, zoned-residential land" -- NOT occupied homes. This bucket is
    # unimproved land zoned residential, or structures the assessor treats
    # as having zero/deficient improvement value (vacant_land_residential
    # in property_type_group(), see revenue_impact/estimate_lost_revenue.py).
    labels = ["Vacant, zoned-\nresidential land", "Commercial", "Industrial"]
    values = [
        summary["city_n_vacant_land_residential"],
        summary["city_n_commercial"],
        summary["city_n_industrial"],
    ]
    colors = [PRINT_TEAL, PRINT_RED, PRINT_BLACK]
    wedges, _, autotexts = ax.pie(
        values, labels=None, colors=colors, autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10, "fontweight": "bold", "color": "white"},
    )
    ax.legend(
        [f"{l} ({v:,})" for l, v in zip(labels, values)],
        loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8.5, ncol=1,
    )
    ax.set_title(f"Citywide: {summary['city_n_vacant_properties']:,} Vacant/Underused Properties",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    target = FIGURES_DIR / "district3_property_types.png"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def main() -> None:
    if not PARCEL_REVENUE_CSV.exists():
        raise SystemExit(f"{PARCEL_REVENUE_CSV} not found -- run revenue_impact/estimate_lost_revenue.py first")

    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    all_parcels, d3, district_summary = load_data()
    summary = build_summary(all_parcels, d3, district_summary)
    write_csv(d3)
    write_summary_json(summary)

    chart_sales_tax_by_district(district_summary)
    chart_vacancy_duration_citywide(all_parcels, summary)
    chart_property_type_breakdown(summary)

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
