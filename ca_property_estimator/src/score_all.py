"""Score ALL California parcels with empirically-derived land + improvement models.

Two-component decomposition:
  ESTIMATED_VALUE = EST_LAND_VALUE + EST_IMPRV_VALUE

Land Model:
  - Training: recent sales (1-2yr) with VAL_ASSD_LAND and LOT_SIZE_AREA
  - Target: LAND_PER_SQFT = VAL_ASSD_LAND / LOT_SIZE_AREA
  - Features: FIPS_TRACT FE + USE_CODE dummies (statewide)
  - Scoring: predicted_land_per_sqft * LOT_SIZE_AREA

Improvement Model:
  - Training: ALL properties with VAL_ASSD_IMPRV, BUILDING_SQFT, DATE_TRANSFER
  - Target: log(VAL_ASSD_IMPRV / BUILDING_SQFT)
  - Features: FIPS_TRACT FE + YEARS_SINCE_SALE + tract*YEARS interaction
              + USE_CODE dummies + USE_CODE*YEARS interaction (statewide)
  - Multiplier: exp(-(beta1 + gamma_tract + theta_use_type) * t)
  - Scoring: EST_IMPRV_VALUE = VAL_ASSD_IMPRV * multiplier(tract, type, t)

At t=0 the multiplier is always 1.0 by construction.

Usage: python -m src.score_all
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .config import (
    load_config, GLMConfig, NonResGLMConfig,
    classify_property_type, get_model_for_property_type,
    PROPERTY_TYPE_RANGES,
)
from .feature_engineering import engineer_features

logger = logging.getLogger(__name__)

# Maximum plausible property value
MAX_VALUE_CLIP = 500_000_000  # $500M

# Use-code groups for dummies (residential is reference category)
USE_TYPE_DUMMIES = [
    "condo", "multifamily", "commercial", "industrial",
    "agricultural", "vacant_land", "institutional", "recreation",
]


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def load_cached_parcels():
    """Load cached parcel data with engineered features."""
    fe_cache = Path("results/all_parcels_fe.parquet")
    if not fe_cache.exists():
        raise FileNotFoundError(
            f"Cache not found: {fe_cache}\n"
            f"Run the residential pipeline first to generate cached data."
        )
    logger.info("Loading cached parcels from %s", fe_cache)
    df = pd.read_parquet(fe_cache)
    logger.info("  Loaded %d parcels with %d columns", len(df), len(df.columns))
    return df


def ensure_fips_tract(df):
    """Add FIPS_TRACT column if missing."""
    if "FIPS_TRACT" not in df.columns:
        fips = df["FIPS_CODE"].astype(str).str.strip()
        tract = df["CENSUS_TRACT"].astype(str).str.strip()
        df["FIPS_TRACT"] = fips + "_" + tract
    return df


def add_use_type_dummies(df):
    """Add binary USE_CODE dummy columns. Residential is reference."""
    for utype in USE_TYPE_DUMMIES:
        col = f"IS_{utype.upper()}"
        if col not in df.columns:
            df[col] = (df["PROPERTY_TYPE_GROUP"] == utype).astype(np.float64)
    return df


def safe_expm1(log_pred, clip_max=MAX_VALUE_CLIP):
    """Apply expm1 with overflow protection."""
    max_log = np.log1p(clip_max)
    log_clipped = np.clip(log_pred, -max_log, max_log)
    values = np.expm1(log_clipped)
    return np.clip(values, 0, clip_max)


# ═══════════════════════════════════════════════════════════════════
# SPARSE GLM SOLVER & DESIGN MATRIX HELPERS
# ═══════════════════════════════════════════════════════════════════

def fit_sparse_ols(X, y, ridge_frac=0.01):
    """Fit OLS via sparse Cholesky: (X'X + aI)b = X'y."""
    XtX = (X.T @ X).tocsc()
    ridge_alpha = ridge_frac * XtX.diagonal().mean()
    XtX += ridge_alpha * sparse.eye(XtX.shape[0], format="csc")
    Xty = X.T @ y
    beta = spsolve(XtX, Xty)
    return beta


def build_tract_fe_matrix(tract_values, tract_categories, pooled_label="__POOLED__"):
    """Build sparse one-hot matrix for tract fixed effects."""
    n = len(tract_values)
    tract_to_idx = {t: i for i, t in enumerate(tract_categories)}
    pooled_idx = tract_to_idx.get(pooled_label, len(tract_categories) - 1)

    col_indices = np.array(
        [tract_to_idx.get(t, pooled_idx) for t in tract_values],
        dtype=np.int32,
    )
    row_indices = np.arange(n, dtype=np.int32)
    data = np.ones(n, dtype=np.float64)

    return sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n, len(tract_categories)),
    )


# ═══════════════════════════════════════════════════════════════════
# LAND VALUE MODEL
# ═══════════════════════════════════════════════════════════════════

def prepare_land_training(df, max_years=2, min_tract_sales=5):
    """Prepare training data for the land value model.

    Uses recent sales where VAL_ASSD_LAND has been reassessed
    to current market value.

    Returns:
        df_train: filtered DataFrame with LAND_PER_SQFT target
        tract_categories: list of valid tracts + pooled label
    """
    df = df.copy()
    df = ensure_fips_tract(df)

    df["_ASSD_LAND"] = pd.to_numeric(df["VAL_ASSD_LAND"], errors="coerce")
    df["_ASSD_TOTAL"] = pd.to_numeric(df["VAL_ASSD"], errors="coerce")
    df["_SALE_PRICE"] = pd.to_numeric(df["VAL_TRANSFER"], errors="coerce")
    df["_LOT_SIZE"] = pd.to_numeric(df["LOT_SIZE_AREA"], errors="coerce")
    df["_SALE_DATE"] = pd.to_datetime(df["DATE_TRANSFER"], errors="coerce", utc=True)
    df["_YEARS_SINCE"] = (
        (pd.Timestamp.now(tz="UTC") - df["_SALE_DATE"]).dt.days / 365.25
    )

    mask = (
        df["_YEARS_SINCE"].notna()
        & (df["_YEARS_SINCE"] >= 0)
        & (df["_YEARS_SINCE"] <= max_years)
        & (df["_ASSD_LAND"] > 0)
        & (df["_ASSD_TOTAL"] > 0)
        & (df["_SALE_PRICE"] >= 50_000)
        & (df["_SALE_PRICE"] <= 20_000_000)
        & (df["_LOT_SIZE"] > 0)
    )
    df = df[mask].copy()
    logger.info("  Recent sales (<%dyr) with land + sale data: %d", max_years, len(df))

    # Use actual sale price, scaled by the assessor's land/total split.
    # VAL_ASSD_LAND undervalues land (~60% of market), but the assessor's
    # relative split (land vs improvement) is their expertise.
    # MARKET_LAND = SALE_PRICE * (ASSD_LAND / ASSD_TOTAL)
    land_share = (df["_ASSD_LAND"] / df["_ASSD_TOTAL"]).clip(0.05, 0.95)
    df["_MARKET_LAND"] = df["_SALE_PRICE"] * land_share
    logger.info("  Land share of assessed: median=%.3f, mean=%.3f",
                land_share.median(), land_share.mean())
    logger.info("  Market land value: median=$%s, mean=$%s",
                f"{df['_MARKET_LAND'].median():,.0f}",
                f"{df['_MARKET_LAND'].mean():,.0f}")

    df["LAND_PER_SQFT"] = df["_MARKET_LAND"] / df["_LOT_SIZE"]
    df["LOG_LAND_PER_SQFT"] = np.log(df["LAND_PER_SQFT"])

    # Outlier filtering: trim top/bottom 5% within each tract
    # Using transform instead of apply to avoid deprecation warning
    before = len(df)
    lo = df.groupby("FIPS_TRACT")["LOG_LAND_PER_SQFT"].transform(
        lambda x: x.quantile(0.05) if len(x) >= 3 else x.min()
    )
    hi = df.groupby("FIPS_TRACT")["LOG_LAND_PER_SQFT"].transform(
        lambda x: x.quantile(0.95) if len(x) >= 3 else x.max()
    )
    df = df[(df["LOG_LAND_PER_SQFT"] >= lo) & (df["LOG_LAND_PER_SQFT"] <= hi)]
    logger.info("  After outlier trimming: %d (removed %d)", len(df), before - len(df))

    global_lo = df["LOG_LAND_PER_SQFT"].quantile(0.01)
    global_hi = df["LOG_LAND_PER_SQFT"].quantile(0.99)
    df = df[(df["LOG_LAND_PER_SQFT"] >= global_lo) & (df["LOG_LAND_PER_SQFT"] <= global_hi)]
    logger.info("  After global trim (1%%/99%%): %d rows", len(df))
    logger.info("  LAND_PER_SQFT: median=$%.2f, mean=$%.2f, p25=$%.2f, p75=$%.2f",
                df["LAND_PER_SQFT"].median(), df["LAND_PER_SQFT"].mean(),
                df["LAND_PER_SQFT"].quantile(0.25),
                df["LAND_PER_SQFT"].quantile(0.75))

    tract_counts = df["FIPS_TRACT"].value_counts()
    valid_tracts = sorted(tract_counts[tract_counts >= min_tract_sales].index.tolist())
    tract_categories = valid_tracts + ["__POOLED__"]
    n_pooled = (tract_counts < min_tract_sales).sum()
    logger.info("  Tracts: %d valid (>=%d sales), %d pooled",
                len(valid_tracts), min_tract_sales, n_pooled)

    df = add_use_type_dummies(df)
    return df, tract_categories


def train_land_model(df_train, tract_categories):
    """Train land model: LAND_PER_SQFT ~ tract FE + use dummies."""
    logger.info("  Training land value model...")
    n = len(df_train)

    intercept = sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))
    dummy_cols = [f"IS_{u.upper()}" for u in USE_TYPE_DUMMIES]
    use_dummies = sparse.csr_matrix(
        df_train[dummy_cols].values.astype(np.float64)
    )
    tract_fe = build_tract_fe_matrix(
        df_train["FIPS_TRACT"].values, tract_categories
    )

    X = sparse.hstack([intercept, use_dummies, tract_fe], format="csr")
    y = df_train["LOG_LAND_PER_SQFT"].values.astype(np.float64)

    col_names = ["intercept"] + dummy_cols + \
        [f"tract__{t}" for t in tract_categories]

    logger.info("  Design matrix: %d x %d, nnz=%d", X.shape[0], X.shape[1], X.nnz)
    beta = fit_sparse_ols(X, y)

    # Evaluate in dollar-space
    y_pred = X @ beta
    pred_dollars = np.exp(y_pred)
    actual_dollars = np.exp(y)
    residuals_log = y - y_pred
    ss_res = np.sum(residuals_log ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    pct_err = np.abs(pred_dollars - actual_dollars) / np.maximum(actual_dollars, 0.01)
    mdape = float(np.median(pct_err) * 100)
    w20 = float((pct_err < 0.20).mean() * 100)

    logger.info("  Land Model Results (in-sample):")
    logger.info("    R2 (log):  %.4f", r2)
    logger.info("    MdAPE:     %.1f%%", mdape)
    logger.info("    Within 20%%: %.1f%%", w20)
    logger.info("    Intercept: %.4f (=$%.2f/sqft at ref tract)",
                beta[0], np.exp(beta[0]))
    for i, name in enumerate(dummy_cols):
        logger.info("    %-25s %.4f (x%.2f)", name, beta[1 + i],
                    np.exp(beta[1 + i]))

    return {
        "beta": beta,
        "tract_categories": tract_categories,
        "dummy_cols": dummy_cols,
        "col_names": col_names,
        "r2": float(r2),
        "mdape": mdape,
    }


def score_land(df, land_model):
    """Score all parcels with the land model."""
    beta = land_model["beta"]
    tract_categories = land_model["tract_categories"]
    dummy_cols = land_model["dummy_cols"]

    df = ensure_fips_tract(df)
    df = add_use_type_dummies(df)
    n = len(df)

    intercept = sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))
    use_dummies = sparse.csr_matrix(
        df[dummy_cols].values.astype(np.float64)
    )
    tract_fe = build_tract_fe_matrix(
        df["FIPS_TRACT"].values, tract_categories
    )

    X = sparse.hstack([intercept, use_dummies, tract_fe], format="csr")
    if X.shape[1] < len(beta):
        pad = sparse.csr_matrix((n, len(beta) - X.shape[1]))
        X = sparse.hstack([X, pad], format="csr")
    elif X.shape[1] > len(beta):
        X = X[:, :len(beta)]

    log_land_per_sqft = X @ beta
    land_per_sqft = np.exp(log_land_per_sqft)
    land_per_sqft = np.maximum(land_per_sqft, 0)

    lot_size = pd.to_numeric(df["LOT_SIZE_AREA"], errors="coerce").fillna(0).values
    land_value = land_per_sqft * lot_size
    return np.clip(land_value, 0, MAX_VALUE_CLIP)


# ═══════════════════════════════════════════════════════════════════
# IMPROVEMENT VALUE MODEL
#
# log(VAL_ASSD_IMPRV / BUILDING_SQFT) ~
#     intercept
#   + tract_FE                          # location level
#   + beta1 * t                         # statewide time decay
#   + gamma_tract * t                   # tract-specific decay
#   + delta_use_type                    # use-type level (statewide)
#   + theta_use_type * t                # use-type-specific decay
#
# Multiplier for scoring:
#   multiplier(tract, use_type, t) = exp(-(beta1 + gamma_tract + theta_use_type) * t)
#
# At t=0, multiplier = 1.0 by construction.
# EST_IMPRV_VALUE = VAL_ASSD_IMPRV * multiplier
# ═══════════════════════════════════════════════════════════════════

def prepare_imprv_training(df, min_tract_obs=10):
    """Prepare training data for the improvement value model."""
    df = df.copy()
    df = ensure_fips_tract(df)

    df["_ASSD_IMPRV"] = pd.to_numeric(df["VAL_ASSD_IMPRV"], errors="coerce")
    df["_ASSD_TOTAL"] = pd.to_numeric(df["VAL_ASSD"], errors="coerce")
    df["_SALE_PRICE"] = pd.to_numeric(df["VAL_TRANSFER"], errors="coerce")
    df["_BLDG_SQFT"] = pd.to_numeric(df["BUILDING_SQFT"], errors="coerce")
    df["_SALE_DATE"] = pd.to_datetime(df["DATE_TRANSFER"], errors="coerce", utc=True)
    df["YEARS_SINCE_SALE"] = (
        (pd.Timestamp.now(tz="UTC") - df["_SALE_DATE"]).dt.days / 365.25
    )

    mask = (
        (df["_ASSD_IMPRV"] > 0)
        & (df["_ASSD_TOTAL"] > 0)
        & (df["_SALE_PRICE"] >= 50_000)
        & (df["_SALE_PRICE"] <= 20_000_000)
        & (df["_BLDG_SQFT"] > 0)
        & df["YEARS_SINCE_SALE"].notna()
        & (df["YEARS_SINCE_SALE"] >= 0)
        & (df["YEARS_SINCE_SALE"] <= 50)
    )
    df = df[mask].copy()
    logger.info("  Properties with sale + improvement data: %d", len(df))

    # Use actual sale price, scaled by assessor's improvement share.
    # This anchors improvement values to market, not understated assessments.
    # MARKET_IMPRV = SALE_PRICE * (ASSD_IMPRV / ASSD_TOTAL)
    imprv_share = (df["_ASSD_IMPRV"] / df["_ASSD_TOTAL"]).clip(0.05, 0.95)
    df["_MARKET_IMPRV"] = df["_SALE_PRICE"] * imprv_share
    logger.info("  Imprv share of assessed: median=%.3f, mean=%.3f",
                imprv_share.median(), imprv_share.mean())
    logger.info("  Market imprv value: median=$%s, mean=$%s",
                f"{df['_MARKET_IMPRV'].median():,.0f}",
                f"{df['_MARKET_IMPRV'].mean():,.0f}")

    df["IMPRV_PER_SQFT"] = df["_MARKET_IMPRV"] / df["_BLDG_SQFT"]
    df["LOG_IMPRV_PER_SQFT"] = np.log(df["IMPRV_PER_SQFT"])

    # Outlier filtering: trim top/bottom 5% within each tract
    before = len(df)
    lo = df.groupby("FIPS_TRACT")["LOG_IMPRV_PER_SQFT"].transform(
        lambda x: x.quantile(0.05) if len(x) >= 3 else x.min()
    )
    hi = df.groupby("FIPS_TRACT")["LOG_IMPRV_PER_SQFT"].transform(
        lambda x: x.quantile(0.95) if len(x) >= 3 else x.max()
    )
    df = df[(df["LOG_IMPRV_PER_SQFT"] >= lo) & (df["LOG_IMPRV_PER_SQFT"] <= hi)]
    logger.info("  After outlier trimming: %d (removed %d)",
                len(df), before - len(df))

    global_lo = df["LOG_IMPRV_PER_SQFT"].quantile(0.01)
    global_hi = df["LOG_IMPRV_PER_SQFT"].quantile(0.99)
    df = df[
        (df["LOG_IMPRV_PER_SQFT"] >= global_lo)
        & (df["LOG_IMPRV_PER_SQFT"] <= global_hi)
    ]
    logger.info("  After global trim: %d rows", len(df))

    imprv = df["IMPRV_PER_SQFT"]
    logger.info("  IMPRV_PER_SQFT: median=$%.2f, mean=$%.2f, p25=$%.2f, p75=$%.2f",
                imprv.median(), imprv.mean(),
                imprv.quantile(0.25), imprv.quantile(0.75))

    tract_counts = df["FIPS_TRACT"].value_counts()
    valid_tracts = sorted(
        tract_counts[tract_counts >= min_tract_obs].index.tolist()
    )
    tract_categories = valid_tracts + ["__POOLED__"]
    n_pooled = (tract_counts < min_tract_obs).sum()
    logger.info("  Tracts: %d valid (>=%d obs), %d pooled",
                len(valid_tracts), min_tract_obs, n_pooled)

    df = add_use_type_dummies(df)
    return df, tract_categories


def train_imprv_model(df_train, tract_categories):
    """Train the improvement value model.

    Design matrix blocks:
      0: intercept
      1: use_type dummies (statewide level)
      2: tract FE (location level)
      3: YEARS_SINCE_SALE (global time slope)
      4: tract * YEARS (tract-specific time slope)
      5: use_type * YEARS (type-specific time slope, statewide)

    Returns dict with model artifacts.
    """
    logger.info("  Training improvement value model...")
    n = len(df_train)
    n_tracts = len(tract_categories)
    dummy_cols = [f"IS_{u.upper()}" for u in USE_TYPE_DUMMIES]
    n_dummies = len(dummy_cols)

    # Block 0: Intercept
    intercept = sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))

    # Block 1: Use-type dummies
    use_dummies = sparse.csr_matrix(
        df_train[dummy_cols].values.astype(np.float64)
    )

    # Block 2: Tract FE
    tract_fe = build_tract_fe_matrix(
        df_train["FIPS_TRACT"].values, tract_categories
    )

    # Block 3: YEARS_SINCE_SALE (global slope)
    years = df_train["YEARS_SINCE_SALE"].values.astype(np.float64)
    years_col = sparse.csr_matrix(years.reshape(-1, 1))

    # Block 4: tract * YEARS interaction
    tract_x_years = tract_fe.multiply(years.reshape(-1, 1))

    # Block 5: use_type * YEARS interaction (statewide type-specific decay)
    use_x_years = sparse.csr_matrix(
        df_train[dummy_cols].values.astype(np.float64)
        * years.reshape(-1, 1)
    )

    X = sparse.hstack(
        [intercept, use_dummies, tract_fe, years_col, tract_x_years, use_x_years],
        format="csr",
    )
    y = df_train["LOG_IMPRV_PER_SQFT"].values.astype(np.float64)

    col_names = (
        ["intercept"]
        + dummy_cols
        + [f"tract__{t}" for t in tract_categories]
        + ["YEARS_SINCE_SALE"]
        + [f"tract_x_years__{t}" for t in tract_categories]
        + [f"{d}_x_YEARS" for d in dummy_cols]
    )

    logger.info("  Design matrix: %d x %d, nnz=%d (%.1f MB)",
                X.shape[0], X.shape[1], X.nnz,
                (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6)

    beta = fit_sparse_ols(X, y)

    # Evaluate (in-sample)
    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    pred_dollars = np.exp(y_pred)
    actual_dollars = np.exp(y)
    pct_err = np.abs(pred_dollars - actual_dollars) / np.maximum(actual_dollars, 0.01)
    mdape = float(np.median(pct_err) * 100)
    w20 = float((pct_err < 0.20).mean() * 100)

    # Extract key coefficients
    # beta1 = global YEARS slope (index: 1 + n_dummies + n_tracts)
    beta1_idx = 1 + n_dummies + n_tracts
    beta1 = beta[beta1_idx]

    # theta = use_type * YEARS slopes (last n_dummies entries)
    theta_start = beta1_idx + 1 + n_tracts
    theta = beta[theta_start:theta_start + n_dummies]

    logger.info("  Improvement Model Results (in-sample):")
    logger.info("    R2:           %.4f", r2)
    logger.info("    MdAPE:        %.1f%%", mdape)
    logger.info("    Within 20%%:  %.1f%%", w20)
    logger.info("    Intercept:    %.4f (=$%.2f/sqft at t=0, pooled tract)",
                beta[0], np.exp(beta[0]))
    logger.info("    YEARS_SINCE_SALE (global): %.4f", beta1)
    logger.info("    Use-type level & decay:")
    for i, name in enumerate(dummy_cols):
        logger.info("      %-25s level=%.4f  decay=%.4f",
                    name, beta[1 + i], theta[i])

    return {
        "beta": beta,
        "tract_categories": tract_categories,
        "dummy_cols": dummy_cols,
        "col_names": col_names,
        "n_tracts": n_tracts,
        "n_dummies": n_dummies,
        "r2": float(r2),
        "mdape": mdape,
    }


def build_multiplier_table(imprv_model, max_years=50):
    """Build a lookup table: tract x use_type x years -> multiplier.

    multiplier(tract, use_type, t) = exp(-(beta1 + gamma_tract + theta_use_type) * t)

    At t=0, multiplier = 1.0 for all combinations.

    Returns:
        DataFrame with columns: FIPS_TRACT, PROPERTY_TYPE_GROUP,
                                YEARS_SINCE_SALE, MULTIPLIER
    """
    beta = imprv_model["beta"]
    tract_categories = imprv_model["tract_categories"]
    dummy_cols = imprv_model["dummy_cols"]
    n_tracts = imprv_model["n_tracts"]
    n_dummies = imprv_model["n_dummies"]

    # Extract slopes
    # beta1 = global YEARS slope
    beta1_idx = 1 + n_dummies + n_tracts
    beta1 = beta[beta1_idx]

    # gamma_tract = tract-specific YEARS slopes
    gamma_start = beta1_idx + 1
    gamma = beta[gamma_start:gamma_start + n_tracts]

    # theta = use_type-specific YEARS slopes
    theta_start = gamma_start + n_tracts
    theta = beta[theta_start:theta_start + n_dummies]

    # Map use-type dummies back to property type names
    # residential is reference (theta=0)
    use_types = ["residential"] + USE_TYPE_DUMMIES
    theta_by_type = {"residential": 0.0}
    for i, utype in enumerate(USE_TYPE_DUMMIES):
        theta_by_type[utype] = theta[i]

    # Build table
    years_range = np.arange(0, max_years + 1, dtype=np.float64)
    rows = []

    for tract_idx, tract in enumerate(tract_categories):
        gamma_t = gamma[tract_idx] if tract_idx < len(gamma) else 0.0
        total_tract_slope = beta1 + gamma_t

        for utype in use_types:
            total_slope = total_tract_slope + theta_by_type[utype]
            # multiplier = exp(-total_slope * t)
            # Negative slope means assessed values decay below market,
            # so multiplier should be > 1 for t > 0
            multipliers = np.exp(-total_slope * years_range)
            for yr, mult in zip(years_range, multipliers):
                rows.append({
                    "FIPS_TRACT": tract,
                    "PROPERTY_TYPE_GROUP": utype,
                    "YEARS_SINCE_SALE": int(yr),
                    "MULTIPLIER": float(mult),
                })

    table = pd.DataFrame(rows)
    logger.info("  Multiplier table: %d rows (%d tracts x %d types x %d years)",
                len(table), len(tract_categories), len(use_types),
                len(years_range))

    # Sanity check: at t=0, all multipliers should be 1.0
    t0 = table[table["YEARS_SINCE_SALE"] == 0]["MULTIPLIER"]
    logger.info("  At t=0: min=%.4f, max=%.4f (should be 1.0)",
                t0.min(), t0.max())

    # Log some example multipliers
    pooled = table[table["FIPS_TRACT"] == "__POOLED__"]
    for utype in ["residential", "commercial", "industrial", "multifamily"]:
        sub = pooled[pooled["PROPERTY_TYPE_GROUP"] == utype]
        if len(sub) > 0:
            m10 = sub[sub["YEARS_SINCE_SALE"] == 10]["MULTIPLIER"].values
            m20 = sub[sub["YEARS_SINCE_SALE"] == 20]["MULTIPLIER"].values
            if len(m10) > 0 and len(m20) > 0:
                logger.info("  Pooled %-15s t=10: %.2fx  t=20: %.2fx",
                            utype, m10[0], m20[0])

    return table




def score_imprv(df, imprv_model):
    """Score all parcels with the improvement model.


    Predicts current market improvement value by setting
    YEARS_SINCE_SALE = 0 in the model:
      EST_IMPRV = exp(intercept + tract_FE + use_dummy) * BUILDING_SQFT




    The time terms (global slope, tract*years, use*years) all zero out
    at t=0, giving us the current market level per sqft.
    """

    beta = imprv_model["beta"]
    tract_categories = imprv_model["tract_categories"]
    dummy_cols = imprv_model["dummy_cols"]
    n_tracts = imprv_model["n_tracts"]

    df = ensure_fips_tract(df)
    df = add_use_type_dummies(df)
    n = len(df)


    # Block 0: Intercept
    intercept = sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))





    # Block 1: Use-type dummies
    use_dummies = sparse.csr_matrix(
        df[dummy_cols].values.astype(np.float64)
    )

















    # Block 2: Tract FE
    tract_fe = build_tract_fe_matrix(
        df["FIPS_TRACT"].values, tract_categories
    )




    # Block 3: YEARS_SINCE_SALE = 0 (predict current market)
    years_col = sparse.csr_matrix(np.zeros((n, 1), dtype=np.float64))







    # Block 4: tract * YEARS = 0
    tract_x_years = sparse.csr_matrix((n, n_tracts), dtype=np.float64)

    # Block 5: use_type * YEARS = 0
    use_x_years = sparse.csr_matrix((n, len(dummy_cols)), dtype=np.float64)

    X = sparse.hstack(
        [intercept, use_dummies, tract_fe, years_col, tract_x_years, use_x_years],
        format="csr",
    )














    # Handle dimension mismatch
    if X.shape[1] < len(beta):
        pad = sparse.csr_matrix((n, len(beta) - X.shape[1]))
        X = sparse.hstack([X, pad], format="csr")
    elif X.shape[1] > len(beta):
        X = X[:, :len(beta)]







    log_imprv_per_sqft = X @ beta
    imprv_per_sqft = np.exp(log_imprv_per_sqft)
    imprv_per_sqft = np.maximum(imprv_per_sqft, 0)



    bldg_sqft = pd.to_numeric(df["BUILDING_SQFT"], errors="coerce").fillna(0).values
    imprv_value = imprv_per_sqft * bldg_sqft

    return np.clip(imprv_value, 0, MAX_VALUE_CLIP)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_score_all():
    """Main entry point — empirical land + improvement models.

    Pipeline:
      1. Load cached parcels
      2. Train land model on recent sales
      3. Train improvement model on all properties
      4. Build multiplier lookup table
      5. Score all parcels (land + improvement)
      6. Write output
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    t_start = time.time()

    # --- Stage 1: Load cached parcels ---
    logger.info("=" * 70)
    logger.info("STAGE 1: Loading cached parcels")
    logger.info("=" * 70)
    df_all = load_cached_parcels()
    df_all = ensure_fips_tract(df_all)

    # --- Stage 2: Train land model ---
    logger.info("=" * 70)
    logger.info("STAGE 2: Training land value model (recent sales)")
    logger.info("=" * 70)
    t0 = time.time()
    df_land_train, land_tracts = prepare_land_training(df_all, max_years=2)
    land_model = train_land_model(df_land_train, land_tracts)
    logger.info("  Land model trained in %ds", time.time() - t0)

    # --- Stage 3: Train improvement model ---
    logger.info("=" * 70)
    logger.info("STAGE 3: Training improvement value model")
    logger.info("=" * 70)
    t0 = time.time()
    df_imprv_train, imprv_tracts = prepare_imprv_training(df_all)
    imprv_model = train_imprv_model(df_imprv_train, imprv_tracts)
    logger.info("  Improvement model trained in %ds", time.time() - t0)

    # --- Stage 4: Build multiplier table ---
    logger.info("=" * 70)
    logger.info("STAGE 4: Building multiplier lookup table")
    logger.info("=" * 70)
    mult_table = build_multiplier_table(imprv_model, max_years=50)

    # --- Stage 5: Score all parcels ---
    logger.info("=" * 70)
    logger.info("STAGE 5: Scoring all parcels")
    logger.info("=" * 70)

    t0 = time.time()
    logger.info("  Scoring land values...")
    est_land = score_land(df_all, land_model)
    logger.info("  Land: median=$%s, mean=$%s",
                f"{np.median(est_land):,.0f}",
                f"{np.mean(est_land):,.0f}")

    logger.info("  Scoring improvement values...")
    est_imprv = score_imprv(df_all, imprv_model)
    logger.info("  Imprv: median=$%s, mean=$%s",
                f"{np.median(est_imprv):,.0f}",
                f"{np.mean(est_imprv):,.0f}")

    est_total = est_land + est_imprv
    est_total = np.clip(est_total, 0, MAX_VALUE_CLIP)
    logger.info("  Total: median=$%s, mean=$%s",
                f"{np.median(est_total):,.0f}",
                f"{np.mean(est_total):,.0f}")
    logger.info("  Scored %d parcels in %ds", len(df_all), time.time() - t0)

    df_all["EST_LAND_VALUE"] = est_land
    df_all["EST_IMPROVEMENT_VALUE"] = est_imprv
    df_all["ESTIMATED_VALUE"] = est_total

    # Model type labels
    df_all["MODEL_TYPE"] = "land_imprv"
    # Parcels with no building get land-only
    bldg_sqft = pd.to_numeric(df_all["BUILDING_SQFT"], errors="coerce").fillna(0)
    land_only_mask = bldg_sqft <= 0
    df_all.loc[land_only_mask, "MODEL_TYPE"] = "land_only"

    # --- Stage 6: Save artifacts and write output ---
    logger.info("=" * 70)
    logger.info("STAGE 6: Saving artifacts and writing output")
    logger.info("=" * 70)

    out_dir = Path("results/model_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(land_model, out_dir / "land_model.joblib")
    joblib.dump(imprv_model, out_dir / "imprv_model.joblib")
    mult_table.to_parquet(out_dir / "multiplier_table.parquet", index=False)
    logger.info("  Saved model artifacts to %s", out_dir)

    # Output columns. VAL_ASSD/_LAND/_IMPRV are already loaded into df_all
    # (used above for training and multiplier computation) -- ship them
    # alongside the estimates so the output is self-sufficient for an
    # estimate-vs-assessed comparison without a separate join against a
    # second cache (which is what scripts/export_sacramento_roi.py had to do
    # against results/all_parcels_fe.parquet, a cache this script already
    # has these columns from -- see load_cached_parcels() above).
    out_cols = [
        "PARCEL_APN", "APN_UNFORMATTED", "FIPS_CODE", "COUNTYNAME",
        "SITE_ADDR", "SITE_CITY", "SITE_ZIP",
        "PROPERTY_TYPE_GROUP", "MODEL_TYPE",
        "BUILDING_SQFT", "LOT_SIZE_AREA", "PROPERTY_AGE",
        "ESTIMATED_VALUE", "EST_LAND_VALUE", "EST_IMPROVEMENT_VALUE",
        "VAL_ASSD", "VAL_ASSD_LAND", "VAL_ASSD_IMPRV",
    ]
    available = [c for c in out_cols if c in df_all.columns]
    df_out = df_all[available].copy()

    out_path = Path("results/ca_all_parcels_estimated_values.parquet")
    df_out.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df_out), out_path)

    # Summary statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for mt in ["land_imprv", "land_only"]:
        sub = df_out[df_out["MODEL_TYPE"] == mt]
        ev = sub["ESTIMATED_VALUE"].dropna()
        if len(ev) > 0:
            logger.info("  %-15s n=%-10d median=$%-15s mean=$%s",
                        mt, len(ev),
                        f"{ev.median():,.0f}",
                        f"{ev.mean():,.0f}")

    # Breakdown by property type
    for ptype in ["residential", "condo", "commercial", "industrial",
                  "multifamily", "agricultural", "vacant_land",
                  "institutional", "recreation", "unknown"]:
        sub = df_out[df_out["PROPERTY_TYPE_GROUP"] == ptype]
        ev = sub["ESTIMATED_VALUE"].dropna()
        if len(ev) > 0:
            land_med = sub["EST_LAND_VALUE"].median() if "EST_LAND_VALUE" in sub else 0
            imprv_med = sub["EST_IMPROVEMENT_VALUE"].median() if "EST_IMPROVEMENT_VALUE" in sub else 0
            logger.info("    %-18s n=%-8d total=$%-12s land=$%-12s imprv=$%s",
                        ptype, len(ev),
                        f"{ev.median():,.0f}",
                        f"{land_med:,.0f}",
                        f"{imprv_med:,.0f}")

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("Pipeline complete in %.0fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("DONE.")

    return land_model, imprv_model, mult_table


if __name__ == "__main__":
    run_score_all()
