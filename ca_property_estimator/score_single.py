"""Score a single property by APN and show detailed breakdown."""
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, ".")
from src.config import load_config, GLMConfig
from src.data_loader import get_snowflake_connection
from src.feature_engineering import engineer_features
from src.sparse_features import (
    DesignInfo, build_design_matrix, get_column_names
)

def score_single_apn(apn, config_path=None):
    cfg = load_config(config_path)
    glm_cfg = GLMConfig()

    # Load model artifacts
    coefs = np.load("results/glm_artifacts/glm_coefficients.npy")
    design_info = joblib.load("results/glm_artifacts/design_info.joblib")

    # Query the property from Snowflake
    conn = get_snowflake_connection(cfg)
    cur = conn.cursor()
    sf = cfg.snowflake
    dc = cfg.data
    fqn = f"{sf.database}.{sf.schema}.{dc.table}"

    query = f"SELECT * FROM {fqn} WHERE PARCEL_APN = '{apn}' LIMIT 1"
    print(f"Querying {apn} from Snowflake...")
    cur.execute(query)
    df = cur.fetch_pandas_all()
    conn.close()

    if len(df) == 0:
        print(f"ERROR: APN {apn} not found")
        return

    print(f"\n{'='*70}")
    print(f"PROPERTY: {apn}")
    print(f"{'='*70}")

    # Basic info
    row = df.iloc[0]
    print(f"\n--- Address ---")
    for col in ["SITE_ADDR", "SITE_CITY", "SITE_ZIP", "COUNTYNAME"]:
        if col in df.columns and pd.notna(row.get(col)):
            print(f"  {col}: {row[col]}")

    print(f"\n--- Physical Characteristics ---")
    for col in ["BUILDING_SQFT", "LIVING_SQFT", "LOT_SIZE_AREA",
                 "BEDROOMS", "TOTAL_BATHS_CALCULATED", "TOTAL_ROOMS",
                 "YR_BLT", "STORIES_NUMBER", "BUILDING_TYPE_SIMPLIFIED",
                 "CONSTRUCTION_CODE_DESC", "AIR_CONDITIONING_TYPE_DESC",
                 "USE_CODE_STD_DESC_LPS"]:
        if col in df.columns and pd.notna(row.get(col)):
            print(f"  {col}: {row[col]}")

    print(f"\n--- Assessment / Sale History ---")
    for col in ["VAL_ASSD", "VAL_ASSD_LAND", "VAL_ASSD_IMPRV",
                 "AVM_VALUE", "VAL_TRANSFER", "DATE_TRANSFER",
                 "PRIOR_SALE_VAL_TRANSFER", "PRIOR_SALE_DATE_TRANSFER"]:
        if col in df.columns and pd.notna(row.get(col)):
            val = row[col]
            if "VAL" in col or "PRICE" in col or "TRANSFER" in col:
                try:
                    val = f"${float(val):,.0f}"
                except (ValueError, TypeError):
                    pass
            print(f"  {col}: {val}")

    print(f"\n--- Location ---")
    for col in ["LATITUDE", "LONGITUDE", "CENSUS_TRACT",
                 "H3_RES7", "H3_RES8"]:
        if col in df.columns and pd.notna(row.get(col)):
            print(f"  {col}: {row[col]}")

    # Feature engineering
    df["SALE_PRICE"] = pd.to_numeric(df["VAL_TRANSFER"], errors="coerce")
    df["SALE_DATE_PARSED"] = pd.to_datetime(
        df["DATE_TRANSFER"], errors="coerce", utc=True
    )
    df["IS_TRAINING"] = False
    df = engineer_features(df)

    # Create unique tract ID: FIPS_CODE + CENSUS_TRACT
    if "FIPS_CODE" in df.columns and "CENSUS_TRACT" in df.columns:
        fips = df["FIPS_CODE"].astype(str).str.strip()
        tract = df["CENSUS_TRACT"].astype(str).str.strip()
        df["CENSUS_TRACT"] = fips + "_" + tract

    print(f"\n--- Engineered Features ---")
    for col in ["PROPERTY_AGE", "YEARS_SINCE_SALE", "BUILDING_SQFT",
                 "LOT_SIZE_AREA", "BEDROOMS", "TOTAL_BATHS_CALCULATED"]:
        if col in df.columns:
            val = df.iloc[0][col]
            print(f"  {col}: {val}")

    # Build design matrix
    X = build_design_matrix(
        df, design_info,
        ac_col=glm_cfg.ac_col,
        construction_col=glm_cfg.construction_col,
        set_years_since_sale_zero=True,
        dtype=np.float32,
    )

    # Predict
    y_pred_log = X.dot(coefs.astype(np.float32))
    est_value = float(np.expm1(y_pred_log[0]))
    est_value_clipped = np.clip(est_value, glm_cfg.min_sale_price, glm_cfg.max_sale_price)

    print(f"\n{'='*70}")
    print(f"MODEL ESTIMATE")
    print(f"{'='*70}")
    print(f"  log(price) prediction: {y_pred_log[0]:.4f}")
    print(f"  Raw estimate: ${est_value:,.0f}")
    print(f"  Clipped estimate: ${est_value_clipped:,.0f}")

    # Decompose: land vs improvement
    assd_total = pd.to_numeric(row.get("VAL_ASSD", 0), errors="coerce")
    assd_land = pd.to_numeric(row.get("VAL_ASSD_LAND", 0), errors="coerce")
    if assd_total > 0:
        land_ratio = np.clip(assd_land / assd_total, 0.05, 0.95)
    else:
        land_ratio = 0.40
    est_land = est_value_clipped * land_ratio
    est_improvement = est_value_clipped * (1 - land_ratio)

    print(f"\n  Land/Improvement Split:")
    print(f"    Assessed land ratio: {land_ratio:.1%}")
    print(f"    Est. land value: ${est_land:,.0f}")
    print(f"    Est. improvement value: ${est_improvement:,.0f}")

    # Show coefficient contributions
    print(f"\n{'='*70}")
    print(f"COEFFICIENT BREAKDOWN")
    print(f"{'='*70}")

    col_names = get_column_names(design_info)
    x_row = X[0].toarray().ravel()
    contributions = x_row * coefs

    # Continuous features
    cont_start, cont_end = design_info.block_ranges["continuous"]
    print(f"\n  Continuous features:")
    for i in range(cont_start, cont_end):
        if abs(contributions[i]) > 1e-6:
            print(f"    {col_names[i]:40s} x_val={x_row[i]:+8.4f}  "
                  f"coef={coefs[i]:+8.4f}  contrib={contributions[i]:+8.4f}")

    # Tract FE
    fe_start, fe_end = design_info.block_ranges["tract_fe"]
    for i in range(fe_start, fe_end):
        if abs(x_row[i]) > 0:
            print(f"\n  Tract fixed effect:")
            print(f"    {col_names[i]:40s} coef={coefs[i]:+8.4f}  "
                  f"(multiplier: {np.exp(coefs[i]):.2f}x)")
            break

    # Tract interactions
    inter_start, inter_end = design_info.block_ranges["tract_interactions"]
    print(f"\n  Tract x Feature interactions:")
    for i in range(inter_start, inter_end):
        if abs(contributions[i]) > 1e-6:
            print(f"    {col_names[i]:40s} x_val={x_row[i]:+8.4f}  "
                  f"coef={coefs[i]:+8.4f}  contrib={contributions[i]:+8.4f}")

    # Global intercept
    intercept_idx = design_info.block_ranges["intercept"][0]
    print(f"\n  Global intercept:")
    print(f"    {'intercept':40s} coef={coefs[intercept_idx]:+8.4f}  "
          f"(base price: ${np.expm1(coefs[intercept_idx]):,.0f})")

    # Total
    total = float(np.sum(contributions[contributions != 0]))
    print(f"\n  Total log(price) = {total:.4f}")
    print(f"  = exp({total:.4f}) - 1 = ${np.expm1(total):,.0f}")

    # Compare to actual sale if available
    sale_price = pd.to_numeric(row.get("VAL_TRANSFER"), errors="coerce")
    if pd.notna(sale_price) and sale_price > 0:
        print(f"\n{'='*70}")
        print(f"COMPARISON TO LAST SALE")
        print(f"{'='*70}")
        print(f"  Last sale price: ${sale_price:,.0f}")
        print(f"  Sale date: {row.get('DATE_TRANSFER', 'unknown')}")
        print(f"  Model estimate (current): ${est_value_clipped:,.0f}")
        pct_diff = (est_value_clipped - sale_price) / sale_price * 100
        print(f"  Difference: {pct_diff:+.1f}%")


if __name__ == "__main__":
    apn = sys.argv[1] if len(sys.argv) > 1 else "00200820070000"
    score_single_apn(apn)