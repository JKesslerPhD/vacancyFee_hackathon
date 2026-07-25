"""
Hedonic property valuation with census tract fixed effects (v4).

Statsmodels OLS on sparse design matrices with ~22K columns.

Model: log(Sale_Price) ~ α_tract + β₁·SQFT + β₂·LOT_SIZE
                       + γ_tract·SQFT + δ_tract·LOT_SIZE
                       + β₃·BEDROOMS + β₄·BATHS + β₅·AGE
                       + β₆·AC×CLIMATE + β₇·YEARS_SINCE_SALE
                       + construction_dummies + ε

Trained on ALL ~1.2M recent arms-length sales from Snowflake.
YEARS_SINCE_SALE absorbs inflation during training; set to 0 at
prediction time to estimate current market value.

Census tracts provide 7,285 fixed effects with median 102 sales each.
Sparse matrices keep memory under ~500 MB despite 22K+ columns.

CARB AB 2446 Embodied Carbon Program
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .config import load_config, GLMConfig
from .data_loader import get_snowflake_connection, identify_training_set
from .feature_engineering import engineer_features
from .sparse_features import (
    DesignInfo,
    fit_design_info,
    build_design_matrix,
    get_column_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_training_data(cfg):
    """
    Load ALL training data from Snowflake — not a sample.

    Pulls ~1.2M recent arms-length sales with transfer values
    between configured min/max bounds.

    Returns:
        pd.DataFrame with ~1.2M rows, ~100 columns
    """
    conn = get_snowflake_connection(cfg)
    cur = conn.cursor()
    sf = cfg.snowflake
    dc = cfg.data
    fqn = f"{sf.database}.{sf.schema}.{dc.table}"

    query = f"""
    SELECT *
    FROM {fqn}
    WHERE SALE_CODE IN ('F','R','0','*')
      AND TRY_TO_NUMBER(VAL_TRANSFER) BETWEEN {dc.min_sale_value} AND {dc.max_sale_value}
      AND TRY_TO_DATE(DATE_TRANSFER) >= DATEADD(year, -{dc.recent_sale_years}, CURRENT_DATE())
      AND SITE_STATE = '{dc.state_filter}'
      AND DUPLICATE_SUMS = FALSE
    """
    logger.info("Loading ALL training data from Snowflake...")
    logger.info("Query filters: sale_codes F/R/0/*, price $%s-$%s, last %d years",
                f"{dc.min_sale_value:,.0f}", f"{dc.max_sale_value:,.0f}",
                dc.recent_sale_years)
    t0 = time.time()
    cur.execute(query)
    df = cur.fetch_pandas_all()
    conn.close()
    elapsed = time.time() - t0
    logger.info("Loaded %d training rows, %d columns in %.0fs",
                len(df), len(df.columns), elapsed)
    return df


# ═══════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_training_df(df, glm_cfg, allowed_property_types=None):
    """
    Clean and filter the training DataFrame.

    Steps:
      1. Parse SALE_PRICE and SALE_DATE
      2. Run feature engineering (PROPERTY_AGE, YEARS_SINCE_SALE, etc.)
      3. Filter by property type (if specified)
      4. Apply price/sqft filters
      5. Drop rows missing the target

    Args:
        df: Raw DataFrame from Snowflake
        glm_cfg: GLMConfig or NonResGLMConfig
        allowed_property_types: List of property type groups to include
            (e.g., ["residential", "condo"]). None = no type filter.

    Returns:
        Cleaned DataFrame (may be smaller than input)
    """
    logger.info("Preparing training data...")
    n_start = len(df)

    # Parse target
    df["SALE_PRICE"] = pd.to_numeric(df["VAL_TRANSFER"], errors="coerce")
    df["SALE_DATE_PARSED"] = pd.to_datetime(
        df["DATE_TRANSFER"], errors="coerce", utc=True
    )
    df["IS_TRAINING"] = True

    # Feature engineering (PROPERTY_AGE, YEARS_SINCE_SALE, categoricals,
    # property type classification, etc.)
    df = engineer_features(df)

    # Create unique tract ID: FIPS_CODE + CENSUS_TRACT
    # Census tract numbers repeat across counties (e.g., "000501" exists
    # in 12+ counties). The FIPS code makes them globally unique.
    tract_col = getattr(glm_cfg, 'tract_col', 'CENSUS_TRACT')
    if "FIPS_CODE" in df.columns and tract_col in df.columns:
        fips = df["FIPS_CODE"].astype(str).str.strip()
        tract = df[tract_col].astype(str).str.strip()
        df[tract_col] = fips + "_" + tract
        logger.info("  Created unique tract IDs (FIPS_TRACT): %d unique tracts",
                    df[tract_col].nunique())

    # Filter by property type if specified
    if allowed_property_types is not None:
        type_mask = df["PROPERTY_TYPE_GROUP"].isin(allowed_property_types)
        n_before_type = len(df)
        df = df[type_mask].copy()
        logger.info("  Property type filter %s: %d -> %d rows",
                    allowed_property_types, n_before_type, len(df))

    # Apply GLM-specific filters
    mask = (
        df["SALE_PRICE"].notna()
        & (df["SALE_PRICE"] >= glm_cfg.min_sale_price)
        & (df["SALE_PRICE"] <= glm_cfg.max_sale_price)
    )
    # SQFT filter — only apply if min_sqft > 0 (non-res may have land-only parcels)
    if glm_cfg.min_sqft > 0:
        sqft = pd.to_numeric(df.get("BUILDING_SQFT"), errors="coerce")
        sqft_valid = sqft.notna() & (sqft >= glm_cfg.min_sqft) & (sqft <= glm_cfg.max_sqft)
        mask = mask & sqft_valid

    # Must have census tract (for models with tract FE)
    if hasattr(glm_cfg, 'tract_col'):
        mask = mask & df[tract_col].notna()

    df = df[mask].copy()
    logger.info("After filtering: %d → %d rows (dropped %d)",
                n_start, len(df), n_start - len(df))

    return df


# ═══════════════════════════════════════════════════════════════════
# Model Fitting
# ═══════════════════════════════════════════════════════════════════

class OLSResult:
    """
    Lightweight container for sparse OLS results.

    Mimics the statsmodels results interface (params, rsquared, etc.)
    but is computed directly via sparse linear algebra to avoid
    statsmodels' inability to handle scipy sparse exog matrices.

    Solves the normal equations:  (X'X + αI) β = X'y
    using scipy.sparse.linalg.spsolve or lsqr.
    """

    def __init__(self, params, rsquared, rsquared_adj, rmse, n_obs, n_params,
                 y_mean, ss_tot, ss_res, bse=None):
        self.params = params
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj
        self.mse_resid = rmse ** 2
        self.rmse = rmse
        self.n_obs = n_obs
        self.n_params = n_params
        self.bse = bse  # standard errors (optional)


def fit_glm(X_train, y_train, column_names):
    """
    Fit OLS on a sparse design matrix via the normal equations.

    Solves: (X'X + αI) β = X'y  using sparse Cholesky / lsqr.

    A tiny Ridge penalty (α=1e-6) ensures numerical stability for
    the ~22K parameter system without materially affecting estimates.

    No separate intercept is added — the tract fixed effects serve
    as group-level intercepts (every row maps to exactly one tract).

    Args:
        X_train: scipy.sparse.csr_matrix, shape (n_samples, n_features)
        y_train: np.ndarray, shape (n_samples,) — log(sale_price)
        column_names: list of str, length n_features

    Returns:
        OLSResult with .params, .rsquared, etc.
    """
    from scipy.sparse.linalg import spsolve

    n, p = X_train.shape
    logger.info(
        "Fitting OLS (sparse direct solve): "
        "%d observations x %d parameters",
        n, p,
    )
    logger.info("  Observations per parameter: %.1f", n / max(p, 1))

    # Convert to float64 for numerical precision
    X = X_train.astype(np.float64).tocsc()
    y = y_train.astype(np.float64)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))

    # ── Build and solve normal equations: (X'X + aI) b = X'y ──
    # Direct closed-form OLS — no iterations needed.
    # Tiny Ridge penalty ensures X'X is positive definite.
    t0 = time.time()

    logger.info("  Computing X'X (%d x %d)...", p, p)
    XtX = X.T.dot(X)  # p x p sparse symmetric

    # Ridge penalty: alpha relative to mean diagonal of X'X
    # With 7K tracts, ~246 have only 10 sales. Without regularization
    # these tracts get extreme coefficients. alpha=0.01 is enough to
    # stabilize without biasing the well-sampled tracts.
    diag_mean = float(XtX.diagonal().mean())
    alpha = 0.01 * diag_mean
    XtX = XtX + sparse.eye(p, format="csc") * alpha
    logger.info("  Ridge alpha = %.2e (0.01 x diag mean %.2e)", alpha, diag_mean)

    logger.info("  Computing X'y...")
    Xty = X.T.dot(y)

    logger.info("  Solving via sparse direct solver (Cholesky)...")
    params = spsolve(XtX, Xty)

    elapsed = time.time() - t0
    logger.info("  Direct solve completed in %.1fs", elapsed)

    # ── Compute diagnostics ──────────────────────────────────
    y_pred = X_train.dot(params)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    rmse = float(np.sqrt(ss_res / max(n - p, 1)))

    logger.info("  R²     = %.4f", r2)
    logger.info("  Adj R² = %.4f", r2_adj)
    logger.info("  RMSE (log-scale) = %.4f", rmse)
    logger.info("  Non-zero coefficients: %d / %d",
                int(np.sum(np.abs(params) > 1e-10)), p)

    return OLSResult(
        params=params,
        rsquared=r2,
        rsquared_adj=r2_adj,
        rmse=rmse,
        n_obs=n,
        n_params=p,
        y_mean=y_mean,
        ss_tot=ss_tot,
        ss_res=ss_res,
    )


def evaluate_model(results, X_test, y_test, design_info):
    """
    Evaluate the fitted model on held-out test data.

    Computes metrics in both log-space and dollar-space.

    Returns:
        dict of metric_name → value
    """
    y_pred_log = X_test.dot(results.params)

    # Log-space metrics
    resid = y_test - y_pred_log
    rmse_log = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Dollar-space metrics
    y_actual = np.expm1(y_test)
    y_pred_dollars = np.maximum(np.expm1(y_pred_log), 0)
    abs_errors = np.abs(y_actual - y_pred_dollars)
    pct_errors = abs_errors / np.maximum(y_actual, 1)

    metrics = {
        "test_r2": float(r2),
        "test_rmse_log": rmse_log,
        "test_rmse_dollars": float(np.sqrt(np.mean((y_actual - y_pred_dollars) ** 2))),
        "test_mae_dollars": float(np.mean(abs_errors)),
        "test_median_ae_dollars": float(np.median(abs_errors)),
        "test_mape_pct": float(np.median(pct_errors) * 100),
        "test_within_10pct": float(np.mean(pct_errors < 0.10) * 100),
        "test_within_20pct": float(np.mean(pct_errors < 0.20) * 100),
        "n_test": len(y_test),
        "n_parameters": design_info.n_cols,
    }

    logger.info("Test set evaluation (n=%d):", len(y_test))
    logger.info("  R² = %.4f", metrics["test_r2"])
    logger.info("  RMSE = $%s", f"{metrics['test_rmse_dollars']:,.0f}")
    logger.info("  MAE  = $%s", f"{metrics['test_mae_dollars']:,.0f}")
    logger.info("  MdAE = $%s", f"{metrics['test_median_ae_dollars']:,.0f}")
    logger.info("  MAPE = %.1f%%", metrics["test_mape_pct"])
    logger.info("  Within 10%%: %.1f%%", metrics["test_within_10pct"])
    logger.info("  Within 20%%: %.1f%%", metrics["test_within_20pct"])

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Coefficient Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_coefficients(results, design_info, column_names):
    """
    Extract and log the most important coefficients.

    Groups results by block (continuous, tract FE, interactions, etc.)
    for interpretability.

    Returns:
        dict with coefficient summaries
    """
    params = results.params
    coef_summary = {}

    # --- Continuous (global) coefficients ---
    cont_start, cont_end = design_info.block_ranges["continuous"]
    cont_coefs = {}
    for i, col in enumerate(design_info.continuous_cols):
        idx = cont_start + i
        cont_coefs[col] = {
            "coefficient": float(params[idx]),
        }
        # Add std errors if available
        if hasattr(results, "bse") and results.bse is not None:
            cont_coefs[col]["std_error"] = float(results.bse[idx])
            cont_coefs[col]["t_stat"] = float(params[idx] / max(results.bse[idx], 1e-10))

    coef_summary["continuous"] = cont_coefs
    logger.info("Global continuous coefficients:")
    for col, vals in cont_coefs.items():
        coef_str = f"  {col}: β = {vals['coefficient']:.4f}"
        if "t_stat" in vals:
            coef_str += f" (t = {vals['t_stat']:.1f})"
        logger.info(coef_str)

    # --- Tract FE summary statistics ---
    fe_start, fe_end = design_info.block_ranges["tract_fe"]
    tract_fes = params[fe_start:fe_end]
    coef_summary["tract_fe_stats"] = {
        "count": len(tract_fes),
        "mean": float(np.mean(tract_fes)),
        "std": float(np.std(tract_fes)),
        "min": float(np.min(tract_fes)),
        "p25": float(np.percentile(tract_fes, 25)),
        "median": float(np.median(tract_fes)),
        "p75": float(np.percentile(tract_fes, 75)),
        "max": float(np.max(tract_fes)),
    }
    logger.info(
        "Tract FE (n=%d): mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
        len(tract_fes), np.mean(tract_fes), np.std(tract_fes),
        np.min(tract_fes), np.max(tract_fes),
    )
    # Implied price multiplier range
    logger.info(
        "  Implied price multiplier range: %.1fx to %.1fx",
        np.exp(np.min(tract_fes)), np.exp(np.max(tract_fes)),
    )

    # --- Tract interaction summary ---
    inter_start, inter_end = design_info.block_ranges["tract_interactions"]
    n_tracts = len(design_info.tract_categories)
    for feat_idx, feat_col in enumerate(design_info.tract_interaction_features):
        offset = inter_start + feat_idx * n_tracts
        slopes = params[offset:offset + n_tracts]
        coef_summary[f"tract_x_{feat_col}_stats"] = {
            "mean": float(np.mean(slopes)),
            "std": float(np.std(slopes)),
            "min": float(np.min(slopes)),
            "median": float(np.median(slopes)),
            "max": float(np.max(slopes)),
        }
        logger.info(
            "Tract × %s slopes: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
            feat_col, np.mean(slopes), np.std(slopes),
            np.min(slopes), np.max(slopes),
        )

    # --- Global intercept ---
    intercept_idx = design_info.block_ranges["intercept"][0]
    intercept_val = float(params[intercept_idx])
    coef_summary["intercept"] = intercept_val
    logger.info("Global intercept: %.4f (base price: $%s)",
                intercept_val, f"{np.expm1(intercept_val):,.0f}")

    return coef_summary


# ═══════════════════════════════════════════════════════════════════
# Tract FE Extraction (for Stage 2 non-residential models)
# ═══════════════════════════════════════════════════════════════════

def extract_tract_fe(results, design_info):
    """
    Extract residential tract fixed effects as a lookup dict.

    Stage 2 non-residential models use these as a continuous
    "location quality" feature rather than fitting their own
    tract FE (too sparse: commercial averages 3.6 sales/tract).

    The tract FE coefficient represents the log-price premium/discount
    of that location relative to the global intercept. A coefficient
    of +1.0 means that tract's properties sell for exp(1) = 2.7x
    the statewide baseline, all else equal.

    Returns:
        dict mapping tract_id (str) -> fe_coefficient (float)
        The pooled tract value is stored under key '__POOLED__'
    """
    params = results.params
    fe_start, fe_end = design_info.block_ranges["tract_fe"]
    tract_fes = params[fe_start:fe_end]

    tract_fe_dict = {}
    for i, tract_id in enumerate(design_info.tract_categories):
        tract_fe_dict[tract_id] = float(tract_fes[i])

    logger.info(
        "Extracted %d tract FE coefficients for Stage 2 "
        "(range: [%.3f, %.3f], median: %.3f)",
        len(tract_fe_dict),
        min(tract_fe_dict.values()),
        max(tract_fe_dict.values()),
        float(np.median(list(tract_fe_dict.values()))),
    )
    return tract_fe_dict


def apply_tract_fe_to_df(df, tract_fe_dict, tract_col="CENSUS_TRACT"):
    """
    Add RES_TRACT_FE column to a DataFrame using extracted tract FE.

    For non-residential parcels, this continuous feature captures
    location quality derived from residential sales — the opportunity
    cost of land in that census tract.

    Parcels in tracts not seen during residential training get the
    pooled tract value.

    Args:
        df: DataFrame with tract_col containing FIPS_TRACT IDs
        tract_fe_dict: dict from extract_tract_fe()
        tract_col: column name with FIPS+TRACT composite keys

    Returns:
        DataFrame with RES_TRACT_FE column added
    """
    pooled_val = tract_fe_dict.get("__POOLED__", 0.0)
    df["RES_TRACT_FE"] = df[tract_col].map(tract_fe_dict).fillna(pooled_val)
    n_mapped = (df["RES_TRACT_FE"] != pooled_val).sum()
    logger.info(
        "Applied residential tract FE: %d/%d parcels mapped (%.1f%%), "
        "%d defaulted to pooled value %.3f",
        n_mapped, len(df), 100 * n_mapped / max(len(df), 1),
        len(df) - n_mapped, pooled_val,
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(config_path=None):
    """
    Train the hedonic GLM on ALL recent sales with census tract FEs.

    Pipeline steps:
      1. Load ~1.2M training sales from Snowflake
      2. Engineer features (PROPERTY_AGE, YEARS_SINCE_SALE, etc.)
      3. Fit DesignInfo (determine tract categories, construction dummies)
      4. Build sparse design matrix (~22K cols, ~150 MB)
      5. Train/test split
      6. Fit statsmodels OLS
      7. Evaluate on held-out test set
      8. Score training set with YEARS_SINCE_SALE=0 for current values
      9. Validate against recent sales
     10. Decompose into land + improvement using assessed ratios
     11. Save model artifacts
     12. Generate report
    """
    cfg = load_config(config_path)
    glm_cfg = GLMConfig()
    glm_cfg.output_dir = Path(cfg.output.directory)
    glm_cfg.model_artifact_dir = Path(cfg.output.directory) / "glm_artifacts"
    glm_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    glm_cfg.model_artifact_dir.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()

    # ── Step 1: Load ALL training data ────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 1: Loading training data from Snowflake")
    logger.info("═" * 60)
    df = load_training_data(cfg)

    # ── Step 2: Feature engineering ───────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 2: Feature engineering")
    logger.info("═" * 60)
    # Filter to residential + condo for Stage 1 model
    # (condos have similar features: beds, baths, sqft)
    df = prepare_training_df(
        df, glm_cfg,
        allowed_property_types=["residential", "condo"],
    )
    logger.info("Training set: %d rows", len(df))

    # Log data coverage
    for col in glm_cfg.continuous_features + [glm_cfg.tract_col, glm_cfg.construction_col]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            logger.info("  %s: %.1f%% non-null", col, pct)

    # ── Step 3: Fit DesignInfo ────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 3: Fitting design matrix layout")
    logger.info("═" * 60)
    design_info = fit_design_info(
        df=df,
        continuous_cols=glm_cfg.continuous_features,
        tract_col=glm_cfg.tract_col,
        tract_interaction_features=glm_cfg.tract_interaction_features,
        construction_col=glm_cfg.construction_col,
        ac_col=glm_cfg.ac_col,
        min_tract_sales=glm_cfg.min_tract_sales,
    )
    column_names = get_column_names(design_info)

    # ── Step 4: Build sparse design matrix ────────────────────
    logger.info("═" * 60)
    logger.info("STEP 4: Building sparse design matrix")
    logger.info("═" * 60)
    X_all = build_design_matrix(
        df, design_info,
        ac_col=glm_cfg.ac_col,
        construction_col=glm_cfg.construction_col,
        set_years_since_sale_zero=False,  # Training: use actual values
        dtype=np.float32,
    )
    y_all = np.log1p(df["SALE_PRICE"].values).astype(np.float32)
    logger.info("Target: log1p(SALE_PRICE), range [%.2f, %.2f]",
                y_all.min(), y_all.max())

    # ── Step 5: Train/test split ──────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 5: Train/test split")
    logger.info("═" * 60)
    from sklearn.model_selection import train_test_split

    test_size = cfg.model.test_size
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=cfg.model.random_state,
    )
    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]
    logger.info("Train: %d rows, Test: %d rows", len(idx_train), len(idx_test))

    # Free some memory
    del X_all
    gc.collect()

    # ── Step 6: Fit OLS ───────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 6: Fitting statsmodels OLS")
    logger.info("═" * 60)
    ols_results = fit_glm(X_train, y_train, column_names)

    # Free training matrices
    del X_train, y_train
    gc.collect()

    # ── Step 7: Evaluate on test set ──────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 7: Evaluating on test set")
    logger.info("═" * 60)
    test_metrics = evaluate_model(ols_results, X_test, y_test, design_info)

    del X_test, y_test
    gc.collect()

    # ── Step 8: Analyze coefficients ──────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 8: Coefficient analysis")
    logger.info("═" * 60)
    coef_summary = analyze_coefficients(ols_results, design_info, column_names)

    # ── Step 8b: Extract tract FE for Stage 2 non-res models ──
    logger.info("Extracting tract FE for non-residential Stage 2 models...")
    tract_fe_dict = extract_tract_fe(ols_results, design_info)

    # ── Step 9: Score training set (current values) ───────────
    logger.info("═" * 60)
    logger.info("STEP 9: Scoring training set with YEARS_SINCE_SALE=0")
    logger.info("═" * 60)
    # Rebuild design matrix with YEARS_SINCE_SALE=0 for current values
    X_current = build_design_matrix(
        df, design_info,
        ac_col=glm_cfg.ac_col,
        construction_col=glm_cfg.construction_col,
        set_years_since_sale_zero=True,  # Predict CURRENT value
        dtype=np.float32,
    )
    y_pred_log = X_current.dot(ols_results.params)
    est_values = np.maximum(np.expm1(y_pred_log), 0)

    # Clip to training data range — predictions outside
    # [min_sale, max_sale] are extrapolation artifacts from
    # extreme tract × lot_size combinations
    est_values = np.clip(est_values, glm_cfg.min_sale_price, glm_cfg.max_sale_price)
    df["ESTIMATED_VALUE"] = est_values

    del X_current
    gc.collect()

    logger.info(
        "Current value estimates — median: $%s | mean: $%s | "
        "min: $%s | max: $%s",
        f"{np.median(est_values):,.0f}",
        f"{np.mean(est_values):,.0f}",
        f"{np.min(est_values):,.0f}",
        f"{np.max(est_values):,.0f}",
    )

    # ── Step 10: Validate against recent sales ────────────────
    logger.info("═" * 60)
    logger.info("STEP 10: Validation")
    logger.info("═" * 60)
    _validate_predictions(df)

    # ── Step 11: Land / improvement decomposition ─────────────
    logger.info("═" * 60)
    logger.info("STEP 11: Land/improvement decomposition")
    logger.info("═" * 60)
    df = _decompose_land_improvement(df)

    # ── Step 12: Save artifacts ───────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 12: Saving model artifacts")
    logger.info("═" * 60)
    artifact_dir = glm_cfg.model_artifact_dir
    _save_model_artifacts(
        ols_results, design_info, column_names,
        test_metrics, coef_summary, artifact_dir,
        tract_fe_dict=tract_fe_dict,
    )

    # ── Step 13: Save report ──────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 13: Generating report")
    logger.info("═" * 60)
    report = _build_report(
        test_metrics, coef_summary, design_info,
        n_total=len(df), n_params=design_info.n_cols,
    )
    report_path = glm_cfg.output_dir / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    elapsed_total = time.time() - t_pipeline_start
    logger.info("═" * 60)
    logger.info("Pipeline complete in %.0f seconds (%.1f minutes)",
                elapsed_total, elapsed_total / 60)
    logger.info("═" * 60)

    return ols_results, design_info, df


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _validate_predictions(df):
    """Compare estimated current values to actual sale prices by recency."""
    yss = df["YEARS_SINCE_SALE"]

    # Very recent sales (<1yr): estimate should be close to sale price
    recent_1yr = yss < 1
    if recent_1yr.sum() > 0:
        actual = df.loc[recent_1yr, "SALE_PRICE"]
        estimated = df.loc[recent_1yr, "ESTIMATED_VALUE"]
        pct_errors = np.abs(actual - estimated) / actual
        logger.info(
            "  <1yr sales (n=%d): median error %.1f%%, mean %.1f%%",
            recent_1yr.sum(),
            pct_errors.median() * 100,
            pct_errors.mean() * 100,
        )
        logger.info(
            "    Within 10%%: %.1f%%, within 20%%: %.1f%%",
            (pct_errors < 0.10).mean() * 100,
            (pct_errors < 0.20).mean() * 100,
        )

    # 1-3yr sales: estimate should be slightly higher (appreciation)
    recent_3yr = (yss >= 1) & (yss < 3)
    if recent_3yr.sum() > 0:
        actual = df.loc[recent_3yr, "SALE_PRICE"]
        estimated = df.loc[recent_3yr, "ESTIMATED_VALUE"]
        ratio = estimated / actual
        logger.info(
            "  1-3yr sales (n=%d): median est/sale ratio %.3f "
            "(expect >1 from appreciation)",
            recent_3yr.sum(), ratio.median(),
        )

    # 3-5yr sales: larger appreciation expected
    recent_5yr = (yss >= 3) & (yss <= 5)
    if recent_5yr.sum() > 0:
        actual = df.loc[recent_5yr, "SALE_PRICE"]
        estimated = df.loc[recent_5yr, "ESTIMATED_VALUE"]
        ratio = estimated / actual
        logger.info(
            "  3-5yr sales (n=%d): median est/sale ratio %.3f",
            recent_5yr.sum(), ratio.median(),
        )


def _decompose_land_improvement(df):
    """
    Split estimated total value into land and improvement components
    using the assessed value ratio from Prop 13.

    This is the same approach as v1-v3 — the ratio from the assessor
    is the best available signal for the split, even though the levels
    are stale under Prop 13.
    """
    assd_total = pd.to_numeric(df["VAL_ASSD"], errors="coerce").replace(0, np.nan)
    assd_land = pd.to_numeric(df["VAL_ASSD_LAND"], errors="coerce")
    land_ratio = (assd_land / assd_total).clip(0.05, 0.95).fillna(0.40)

    df["EST_LAND_VALUE"] = df["ESTIMATED_VALUE"] * land_ratio
    df["EST_IMPROVEMENT_VALUE"] = df["ESTIMATED_VALUE"] * (1 - land_ratio)

    logger.info(
        "Land/Improvement — median land: $%s, median improvement: $%s",
        f"{df['EST_LAND_VALUE'].median():,.0f}",
        f"{df['EST_IMPROVEMENT_VALUE'].median():,.0f}",
    )
    logger.info(
        "  Land ratio — median: %.1f%%, mean: %.1f%%",
        land_ratio.median() * 100,
        land_ratio.mean() * 100,
    )
    return df


def _save_model_artifacts(results, design_info, column_names,
                          test_metrics, coef_summary, artifact_dir,
                          tract_fe_dict=None):
    """
    Save everything needed to rebuild predictions on new data.

    Artifacts:
      - coefficients (numpy array)
      - design_info (DesignInfo dataclass)
      - column_names (list)
      - metrics & coefficient summary (JSON)
      - tract_fe_dict (JSON) — for Stage 2 non-res models
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Coefficients
    coef_path = artifact_dir / "glm_coefficients.npy"
    np.save(coef_path, results.params)
    logger.info("Saved %d coefficients to %s",
                len(results.params), coef_path)

    # DesignInfo (everything needed to rebuild the design matrix)
    info_path = artifact_dir / "design_info.joblib"
    joblib.dump(design_info, info_path)
    logger.info("Saved DesignInfo to %s", info_path)

    # Column names
    names_path = artifact_dir / "column_names.json"
    with open(names_path, "w") as f:
        json.dump(column_names, f)

    # Metrics & coefficients
    summary_path = artifact_dir / "model_summary.json"
    summary = {
        "test_metrics": test_metrics,
        "coefficient_summary": coef_summary,
        "n_parameters": design_info.n_cols,
        "model_type": "statsmodels_OLS_sparse",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved model summary to %s", summary_path)

    # Tract FE lookup — used by Stage 2 non-residential models
    # as a continuous "location quality" feature
    if tract_fe_dict is not None:
        fe_path = artifact_dir / "tract_fe_dict.json"
        with open(fe_path, "w") as f:
            json.dump(tract_fe_dict, f)
        logger.info("Saved %d tract FE values to %s",
                    len(tract_fe_dict), fe_path)


def _build_report(test_metrics, coef_summary, design_info,
                  n_total, n_params):
    """Assemble the model report dictionary."""
    return {
        "model_version": "v4_glm_tract_fe",
        "model_type": "statsmodels OLS with census tract fixed effects",
        "specification": (
            "log(Sale_Price) ~ mu + alpha_tract "
            "+ beta·SQFT + beta·LOT_SIZE "
            "+ gamma_tract·SQFT + delta_tract·LOT_SIZE "
            "+ beta·BEDROOMS + beta·BATHS + beta·AGE "
            "+ beta·HAS_AC + beta·YEARS_SINCE_SALE "
            "+ beta·IS_WOOD + beta·IS_CONCRETE + ..."
        ),
        "dataset": {
            "training_observations": n_total,
            "n_parameters": n_params,
            "obs_per_parameter": round(n_total / max(n_params, 1), 1),
            "n_tracts": len(design_info.tract_categories),
            "n_construction_dummies": len(design_info.construction_dummies),
        },
        "test_metrics": test_metrics,
        "coefficient_summary": coef_summary,
    }


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="CA Property Estimator v4 — GLM with Census Tract FEs"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
