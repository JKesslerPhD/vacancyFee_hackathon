"""Generate figures for the Model Methods document.

Reads from the cached parquet files in results/ and produces
PNG figures in docs/figures/.

Usage:
    cd ca_property_estimator
    python docs/generate_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Optional: skip if matplotlib not installed
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — skipping figure generation")
    sys.exit(0)

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the scored parcels and feature-engineered cache."""
    scored = pd.read_parquet("results/ca_all_parcels_estimated_values.parquet")
    print(f"Loaded {len(scored):,} scored parcels")
    return scored


def fig1_assd_sale_ratio(df):
    """Figure 1: ASSD/SALE_PRICE ratio by sale vintage."""
    sale = pd.to_numeric(df.get("VAL_TRANSFER", pd.Series(dtype=float)), errors="coerce")
    assd = pd.to_numeric(df.get("VAL_ASSD", pd.Series(dtype=float)), errors="coerce")
    if "YEARS_SINCE_SALE" not in df.columns:
        print("  Skipping Fig 1: YEARS_SINCE_SALE not in scored output")
        return
    yss = df["YEARS_SINCE_SALE"]
    mask = (sale > 50000) & (assd > 0) & yss.notna() & (yss >= 0)
    ratio = (assd[mask] / sale[mask]).clip(0, 3)
    yss_vals = yss[mask]

    bins = [0, 1, 3, 5, 10, 20, 30, 50]
    labels = ["0-1yr", "1-3yr", "3-5yr", "5-10yr", "10-20yr", "20-30yr", "30+yr"]
    groups = pd.cut(yss_vals, bins=bins, labels=labels)
    medians = ratio.groupby(groups).median()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(medians)), medians.values, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(medians)))
    ax.set_xticklabels(labels, rotation=30)
    ax.axhline(y=1.0, linestyle="--", color="red", alpha=0.7, label="Parity (1.0)")
    ax.set_ylabel("Median ASSD / SALE_PRICE")
    ax.set_xlabel("Years Since Sale")
    ax.set_title("Figure 1: Proposition 13 Assessment Gap\n"
                 "Assessed values systematically understate market value")
    ax.legend()
    ax.set_ylim(0, 1.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_assd_sale_ratio.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_assd_sale_ratio.png")


def fig3_value_distribution(df):
    """Figure 3: Distribution of estimated values by property type."""
    types = ["residential", "condo", "commercial", "industrial",
             "agricultural", "vacant_land", "institutional", "recreation"]
    data = []
    labels = []
    for t in types:
        sub = df[df["PROPERTY_TYPE_GROUP"] == t]["ESTIMATED_VALUE"]
        sub = sub[sub > 0].dropna()
        if len(sub) > 100:
            data.append(np.log10(sub.values))
            labels.append(f"{t}\n(n={len(sub):,})")

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
              "#795548", "#607D8B", "#F44336", "#00BCD4"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Estimated Value (log₁₀ $)")
    ax.set_title("Figure 3: Distribution of Estimated Market Values by Property Type")
    yticks = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${10**y:,.0f}" for y in yticks])
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_value_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_value_distribution.png")


def fig6_tract_fe_distribution():
    """Figure 6: Tract fixed effect distribution (from model artifacts)."""
    import json
    fe_path = Path("results/glm_artifacts/tract_fe_dict.json")
    if not fe_path.exists():
        print("  Skipping Fig 6: tract_fe_dict.json not found")
        return
    with open(fe_path) as f:
        fe_dict = json.load(f)

    values = [v for k, v in fe_dict.items() if k != "__POOLED__"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=80, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, linestyle="--", color="red", alpha=0.7, label="State average")
    ax.set_xlabel("Tract Fixed Effect Coefficient")
    ax.set_ylabel("Number of Tracts")
    ax.set_title(f"Figure 6: Census Tract Fixed Effects (n={len(values):,})\n"
                 f"Range: {min(values):.2f} to {max(values):.2f} "
                 f"(price multiplier: {np.exp(min(values)):.1f}x to "
                 f"{np.exp(max(values)):.1f}x)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_tract_fe_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved fig6_tract_fe_distribution.png")


def main():
    print("Generating Model Methods figures...")
    df = load_data()
    fig1_assd_sale_ratio(df)
    fig3_value_distribution(df)
    fig6_tract_fe_distribution()
    print("Done. Figures saved to docs/figures/")


if __name__ == "__main__":
    main()