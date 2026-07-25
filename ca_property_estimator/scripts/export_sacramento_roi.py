"""Export Sacramento estimated values with ROI calculations."""
import pandas as pd
import numpy as np
import os

print("Loading scored output...")
df = pd.read_parquet("results/ca_all_parcels_estimated_values.parquet")
sac = df[df["COUNTYNAME"] == "SACRAMENTO"].copy()
print(f"Sacramento parcels (scored): {len(sac):,}")

# Load additional columns from FE cache
print("Loading additional columns from FE cache...")
fe_cols = [
    "PARCEL_APN", "VAL_ASSD", "VAL_ASSD_LAND", "VAL_ASSD_IMPRV",
    "VAL_TRANSFER", "DATE_TRANSFER", "YR_BLT",
    "BEDROOMS", "TOTAL_BATHS_CALCULATED",
    "USE_CODE_STD_DESC_LPS", "CENSUS_TRACT", "COUNTYNAME",
]
fe = pd.read_parquet("results/all_parcels_fe.parquet", columns=fe_cols)
fe_sac = fe[fe["COUNTYNAME"] == "SACRAMENTO"].drop(columns=["COUNTYNAME"])
fe_sac = fe_sac.drop_duplicates(subset="PARCEL_APN", keep="first")
print(f"FE cache Sacramento (deduplicated): {len(fe_sac):,}")

# Merge
out = sac.merge(fe_sac, on="PARCEL_APN", how="left")
print(f"After merge: {len(out):,}")

# Parse for ROI calculation
out["_SALE_PRICE"] = pd.to_numeric(out["VAL_TRANSFER"], errors="coerce")
out["_SALE_DATE"] = pd.to_datetime(out["DATE_TRANSFER"], errors="coerce", utc=True)
out["YEARS_SINCE_SALE"] = (
    (pd.Timestamp.now(tz="UTC") - out["_SALE_DATE"]).dt.days / 365.25
)

# === ROI Calculations ===
roi_mask = (
    out["_SALE_PRICE"].notna()
    & (out["_SALE_PRICE"] >= 10_000)
    & (out["_SALE_PRICE"] <= 20_000_000)
    & out["ESTIMATED_VALUE"].notna()
    & (out["ESTIMATED_VALUE"] > 0)
    & out["YEARS_SINCE_SALE"].notna()
    & (out["YEARS_SINCE_SALE"] > 0)
    & (out["YEARS_SINCE_SALE"] <= 50)
)

# Total ROI = (Current Estimate - Purchase Price) / Purchase Price
out["TOTAL_ROI_PCT"] = np.nan
out.loc[roi_mask, "TOTAL_ROI_PCT"] = (
    (out.loc[roi_mask, "ESTIMATED_VALUE"] - out.loc[roi_mask, "_SALE_PRICE"])
    / out.loc[roi_mask, "_SALE_PRICE"]
    * 100
)

# Annualized CAGR = (End/Start)^(1/years) - 1
out["ANNUALIZED_RETURN_PCT"] = np.nan
price_ratio = out.loc[roi_mask, "ESTIMATED_VALUE"] / out.loc[roi_mask, "_SALE_PRICE"]
years = out.loc[roi_mask, "YEARS_SINCE_SALE"]
cagr = (price_ratio ** (1.0 / years) - 1.0) * 100
out.loc[roi_mask, "ANNUALIZED_RETURN_PCT"] = cagr

# Clip extreme returns (data quality issues)
extreme = (
    (out["ANNUALIZED_RETURN_PCT"] > 100)
    | (out["ANNUALIZED_RETURN_PCT"] < -50)
)
out.loc[extreme, "ANNUALIZED_RETURN_PCT"] = np.nan
out.loc[extreme, "TOTAL_ROI_PCT"] = np.nan

# Round
out["TOTAL_ROI_PCT"] = out["TOTAL_ROI_PCT"].round(1)
out["ANNUALIZED_RETURN_PCT"] = out["ANNUALIZED_RETURN_PCT"].round(2)
out["YEARS_SINCE_SALE"] = out["YEARS_SINCE_SALE"].round(1)

# Drop temp columns
out.drop(columns=["_SALE_PRICE", "_SALE_DATE"], inplace=True)

# === Print Summary ===
cleaned = out[out["ANNUALIZED_RETURN_PCT"].notna()]
print()
print("=" * 70)
print("SACRAMENTO COUNTY — ESTIMATED PROPERTY RETURNS")
print("=" * 70)
print(f"Parcels with calculable ROI: {len(cleaned):,} of {len(out):,}")
print()
med_roi = cleaned["TOTAL_ROI_PCT"].median()
mean_roi = cleaned["TOTAL_ROI_PCT"].mean()
med_cagr = cleaned["ANNUALIZED_RETURN_PCT"].median()
mean_cagr = cleaned["ANNUALIZED_RETURN_PCT"].mean()
print(f"  Median total ROI:        {med_roi:+.1f}%")
print(f"  Mean total ROI:          {mean_roi:+.1f}%")
print(f"  Median annualized CAGR:  {med_cagr:+.2f}%/yr")
print(f"  Mean annualized CAGR:    {mean_cagr:+.2f}%/yr")

print("\nBy sale vintage:")
bins = [0, 1, 3, 5, 10, 20, 30, 50]
labels = ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10-20yr", "20-30yr", "30+yr"]
cleaned_copy = cleaned.copy()
cleaned_copy["VINTAGE"] = pd.cut(cleaned_copy["YEARS_SINCE_SALE"], bins=bins, labels=labels)
for label in labels:
    sub = cleaned_copy[cleaned_copy["VINTAGE"] == label]
    if len(sub) > 50:
        print(
            f"  {label:8s}  n={len(sub):>6,}  "
            f"median_ROI={sub['TOTAL_ROI_PCT'].median():>+7.1f}%  "
            f"median_CAGR={sub['ANNUALIZED_RETURN_PCT'].median():>+6.2f}%/yr  "
            f"med_purchase=${sub['VAL_TRANSFER'].apply(pd.to_numeric, errors='coerce').median():>12,.0f}  "
            f"med_current=${sub['ESTIMATED_VALUE'].median():>12,.0f}"
        )

print("\nBy property type:")
for ptype in ["residential", "condo", "commercial", "industrial"]:
    sub = cleaned_copy[cleaned_copy["PROPERTY_TYPE_GROUP"] == ptype]
    if len(sub) > 50:
        print(
            f"  {ptype:15s}  n={len(sub):>6,}  "
            f"median_ROI={sub['TOTAL_ROI_PCT'].median():>+7.1f}%  "
            f"median_CAGR={sub['ANNUALIZED_RETURN_PCT'].median():>+6.2f}%/yr"
        )

# === Select final columns and write ===
out_cols = [
    "PARCEL_APN",
    "SITE_ADDR", "SITE_CITY", "SITE_ZIP",
    "COUNTYNAME", "FIPS_CODE", "CENSUS_TRACT",
    "PROPERTY_TYPE_GROUP", "USE_CODE_STD_DESC_LPS",
    "BUILDING_SQFT", "LOT_SIZE_AREA", "BEDROOMS", "TOTAL_BATHS_CALCULATED",
    "YR_BLT", "PROPERTY_AGE",
    "ESTIMATED_VALUE", "EST_LAND_VALUE", "EST_IMPROVEMENT_VALUE",
    "MODEL_TYPE",
    "VAL_ASSD", "VAL_ASSD_LAND", "VAL_ASSD_IMPRV",
    "VAL_TRANSFER", "DATE_TRANSFER",
    "YEARS_SINCE_SALE",
    "TOTAL_ROI_PCT", "ANNUALIZED_RETURN_PCT",
]
available = [c for c in out_cols if c in out.columns]
out = out[available]

# Round dollar values
for c in ["ESTIMATED_VALUE", "EST_LAND_VALUE", "EST_IMPROVEMENT_VALUE"]:
    out[c] = out[c].round(0)

out_path = "results/sacramento_estimated_values.csv"
out.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"\nWrote {len(out):,} rows x {len(out.columns)} columns ({size_mb:.1f} MB)")
print(f"File: {out_path}")
print(f"\nNew columns added: YEARS_SINCE_SALE, TOTAL_ROI_PCT, ANNUALIZED_RETURN_PCT")