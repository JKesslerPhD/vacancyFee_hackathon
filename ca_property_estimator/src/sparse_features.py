"""
Sparse design matrix builder v2 — correct intercept structure.

Model specification:
    log(Sale_Price) ~ μ (global intercept)
                    + α_tract (tract deviation from μ)
                    + β₁·SQFT + β₂·LOT_SIZE
                    + γ_tract·SQFT + δ_tract·LOT_SIZE
                    + β₃·BEDROOMS + β₄·BATHS + β₅·AGE
                    + β₆·HAS_AC + β₇·YEARS_SINCE_SALE
                    + β₈·IS_WOOD + β₉·IS_CONCRETE + ...
                    + ε

Key change from v1: AC×Climate and Construction are now
CONTINUOUS/BINARY features, not separate intercept blocks.
This lets the tract FEs capture true location value (e.g.,
downtown Sacramento > state average) without being polluted
by housing stock composition.

Design matrix blocks:
  1. Global intercept (column of 1s)
  2. Continuous features (z-scored, including HAS_AC binary)
  3. Tract FE indicators (one-hot, deviations from global μ)
  4. Tract × feature interactions

CARB AB 2446 Embodied Carbon Program
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class DesignInfo:
    """
    Metadata describing the sparse design matrix layout.

    Stored alongside model coefficients so we can rebuild the
    design matrix identically at prediction time.
    """
    # Column name lists (order matches matrix columns)
    continuous_cols: List[str] = field(default_factory=list)
    tract_categories: List[str] = field(default_factory=list)
    tract_interaction_features: List[str] = field(default_factory=list)

    # Column index ranges (start, end) for each block
    block_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    n_cols: int = 0

    # Imputation values (medians from training set)
    impute_medians: Dict[str, float] = field(default_factory=dict)

    # Tract col name and pooling label
    tract_col: str = "CENSUS_TRACT"
    pooled_tract_label: str = "__POOLED__"

    # Standardization parameters (mean, std) for continuous features
    continuous_means: Dict[str, float] = field(default_factory=dict)
    continuous_stds: Dict[str, float] = field(default_factory=dict)

    # Construction type dummies folded into continuous cols
    construction_dummies: List[str] = field(default_factory=list)


def _has_ac(df: pd.DataFrame, ac_col: str) -> pd.Series:
    """Return binary 0/1 for AC presence."""
    if ac_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    ac = df[ac_col].astype(str).str.strip().str.upper()
    has = ~ac.isin(["UNKNOWN", "NONE", "NAN", ""])
    return has.astype(float)


def _make_construction_dummies(df: pd.DataFrame, col: str,
                                valid_categories: List[str]) -> pd.DataFrame:
    """
    Create binary dummy columns for construction types.
    Returns a DataFrame with columns like IS_WOOD, IS_FRAME, etc.
    """
    result = pd.DataFrame(index=df.index)
    if col not in df.columns:
        for cat in valid_categories:
            result[f"IS_{cat}"] = 0.0
        return result

    vals = df[col].astype(str).str.strip().str.upper()
    for cat in valid_categories:
        result[f"IS_{cat}"] = (vals == cat).astype(float)
    return result


def fit_design_info(
    df: pd.DataFrame,
    continuous_cols: List[str],
    tract_col: str,
    tract_interaction_features: List[str],
    construction_col: str,
    ac_col: str,
    min_tract_sales: int = 10,
) -> DesignInfo:
    """
    Analyze training data and build a DesignInfo.

    v2 changes:
      - AC is a binary continuous feature (HAS_AC), not an intercept block
      - Construction types are binary dummies in the continuous block
      - Global intercept column added
      - Tract FEs are deviations from the global intercept
    """
    info = DesignInfo()
    info.tract_col = tract_col
    info.tract_interaction_features = list(tract_interaction_features)

    # --- Construction dummies (determine valid categories) ---
    if construction_col in df.columns:
        const = df[construction_col].astype(str).str.strip().str.upper()
        const = const.replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
        const_counts = const.value_counts()
        # Keep top categories with at least 100 observations, drop the most common
        # (it becomes the reference category to avoid collinearity with intercept)
        valid_const = const_counts[const_counts >= 100].index.tolist()
        if valid_const:
            # Drop the most common as reference
            reference = valid_const[0]  # most common
            valid_const = sorted([c for c in valid_const if c != reference])
            logger.info("Construction dummies: %s (reference: %s)",
                        valid_const, reference)
        info.construction_dummies = valid_const
    else:
        info.construction_dummies = []

    # --- Build full continuous column list ---
    # Original continuous + HAS_AC + construction dummies
    all_continuous = list(continuous_cols) + ["HAS_AC"] + \
                     [f"IS_{c}" for c in info.construction_dummies]
    info.continuous_cols = all_continuous

    # --- Compute continuous feature stats ---
    # First, augment df with the derived columns temporarily
    ac_vals = _has_ac(df, ac_col)
    const_dummies = _make_construction_dummies(
        df, construction_col, info.construction_dummies)

    for col in all_continuous:
        if col == "HAS_AC":
            vals = ac_vals
        elif col.startswith("IS_") and col[3:] in info.construction_dummies:
            vals = const_dummies[col]
        elif col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
        else:
            vals = pd.Series(0.0, index=df.index)

        med = vals.median()
        info.impute_medians[col] = float(med) if pd.notna(med) else 0.0
        mean_val = vals.mean()
        std_val = vals.std()
        info.continuous_means[col] = float(mean_val) if pd.notna(mean_val) else 0.0
        # For binary features (HAS_AC, IS_*), don't standardize — keep 0/1
        if col == "HAS_AC" or col.startswith("IS_"):
            info.continuous_means[col] = 0.0
            info.continuous_stds[col] = 1.0
        else:
            info.continuous_stds[col] = float(std_val) if (pd.notna(std_val) and std_val > 0) else 1.0

    # --- Census tract categories (pool small tracts) ---
    tract_counts = df[tract_col].value_counts()
    valid_tracts = tract_counts[tract_counts >= min_tract_sales].index.tolist()
    valid_tracts = sorted([str(t) for t in valid_tracts])
    info.tract_categories = valid_tracts + [info.pooled_tract_label]
    logger.info(
        "Tracts: %d with >=%d sales, %d pooled -> %d total FE categories",
        len(valid_tracts), min_tract_sales,
        len(tract_counts) - len(valid_tracts),
        len(info.tract_categories),
    )

    # --- Compute column layout ---
    col_idx = 0

    # Block 0: Global intercept
    info.block_ranges["intercept"] = (col_idx, col_idx + 1)
    col_idx += 1

    # Block 1: Continuous features (including HAS_AC and construction dummies)
    info.block_ranges["continuous"] = (col_idx, col_idx + len(info.continuous_cols))
    col_idx += len(info.continuous_cols)

    # Block 2: Tract FE indicators (deviations from global intercept)
    info.block_ranges["tract_fe"] = (col_idx, col_idx + len(info.tract_categories))
    col_idx += len(info.tract_categories)

    # Block 3: Tract x feature interactions
    n_interactions = len(info.tract_categories) * len(info.tract_interaction_features)
    info.block_ranges["tract_interactions"] = (col_idx, col_idx + n_interactions)
    col_idx += n_interactions

    info.n_cols = col_idx
    logger.info(
        "Design matrix layout: %d total columns "
        "(intercept=1, continuous=%d, tract_fe=%d, tract_interactions=%d)",
        info.n_cols,
        len(info.continuous_cols),
        len(info.tract_categories),
        n_interactions,
    )
    return info

def build_design_matrix(
    df: pd.DataFrame,
    info: DesignInfo,
    ac_col: str = "AIR_CONDITIONING_TYPE_DESC",
    construction_col: str = "CONSTRUCTION_CODE_DESC",
    set_years_since_sale_zero: bool = False,
    dtype: np.dtype = np.float32,
) -> sparse.csr_matrix:
    """
    Build a sparse CSR design matrix from a DataFrame using a fitted DesignInfo.

    v2 layout:
      0. Global intercept (1 for every row)
      1. Continuous features (z-scored originals + HAS_AC binary + construction dummies)
      2. Tract FE indicators (one-hot, deviations from global intercept)
      3. Tract x feature interactions

    Returns:
        scipy.sparse.csr_matrix of shape (n_rows, info.n_cols)
    """
    n = len(df)
    # Estimate nnz: 1 intercept + ~8-10 continuous + 1 tract + 2 interactions = ~14
    estimated_nnz = n * (1 + len(info.continuous_cols) + 1 +
                         len(info.tract_interaction_features))
    rows = np.empty(estimated_nnz, dtype=np.int32)
    cols = np.empty(estimated_nnz, dtype=np.int32)
    vals = np.empty(estimated_nnz, dtype=dtype)
    ptr = 0

    def _extend(r, c, v):
        nonlocal rows, cols, vals, ptr
        count = len(r)
        if ptr + count > len(rows):
            new_size = max(len(rows) * 2, ptr + count + n)
            rows = np.resize(rows, new_size)
            cols = np.resize(cols, new_size)
            vals = np.resize(vals, new_size)
        rows[ptr:ptr + count] = r
        cols[ptr:ptr + count] = c
        vals[ptr:ptr + count] = v
        ptr += count

    row_indices = np.arange(n, dtype=np.int32)

    # ── Block 0: Global intercept ─────────────────────────────
    intercept_col = info.block_ranges["intercept"][0]
    _extend(row_indices,
            np.full(n, intercept_col, dtype=np.int32),
            np.ones(n, dtype=dtype))

    # ── Precompute derived columns ────────────────────────────
    ac_vals = _has_ac(df, ac_col)
    const_dummies = _make_construction_dummies(
        df, construction_col, info.construction_dummies)

    # ── Block 1: Continuous features ──────────────────────────
    cont_start = info.block_ranges["continuous"][0]
    for i, col in enumerate(info.continuous_cols):
        # Get raw values
        if col == "HAS_AC":
            raw = ac_vals.values.astype(np.float64)
        elif col.startswith("IS_") and col[3:] in info.construction_dummies:
            raw = const_dummies[col].values.astype(np.float64)
        elif col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
        else:
            raw = np.full(n, np.nan)

        # Override YEARS_SINCE_SALE for current-value prediction
        if set_years_since_sale_zero and col == "YEARS_SINCE_SALE":
            raw = np.zeros(n)

        # Impute NaN with training median
        median_val = info.impute_medians.get(col, 0.0)
        nans = np.isnan(raw)
        raw[nans] = median_val

        # Standardize (binary features have mean=0, std=1 so stay 0/1)
        mean_val = info.continuous_means.get(col, 0.0)
        std_val = info.continuous_stds.get(col, 1.0)
        standardized = ((raw - mean_val) / std_val).astype(dtype)

        # Only store non-zero values
        nz = standardized != 0
        if nz.sum() > 0:
            _extend(row_indices[nz],
                    np.full(nz.sum(), cont_start + i, dtype=np.int32),
                    standardized[nz])

    # ── Block 2: Tract FE indicators ─────────────────────────
    tract_start = info.block_ranges["tract_fe"][0]
    tract_to_idx = {t: i for i, t in enumerate(info.tract_categories)}
    pooled_idx = tract_to_idx[info.pooled_tract_label]

    if info.tract_col in df.columns:
        tract_vals = df[info.tract_col].astype(str).values
    else:
        tract_vals = np.full(n, info.pooled_tract_label)

    tract_col_indices = np.array(
        [tract_to_idx.get(t, pooled_idx) for t in tract_vals],
        dtype=np.int32,
    )
    _extend(row_indices,
            tract_start + tract_col_indices,
            np.ones(n, dtype=dtype))

    # ── Block 3: Tract x feature interactions ─────────────────
    inter_start = info.block_ranges["tract_interactions"][0]
    n_tracts = len(info.tract_categories)

    for feat_idx, feat_col in enumerate(info.tract_interaction_features):
        if feat_col in df.columns:
            raw = pd.to_numeric(df[feat_col], errors="coerce").values.astype(np.float64)
        else:
            raw = np.full(n, np.nan)

        median_val = info.impute_medians.get(feat_col, 0.0)
        nans = np.isnan(raw)
        raw[nans] = median_val

        mean_val = info.continuous_means.get(feat_col, 0.0)
        std_val = info.continuous_stds.get(feat_col, 1.0)
        standardized = ((raw - mean_val) / std_val).astype(dtype)

        col_offsets = inter_start + feat_idx * n_tracts + tract_col_indices
        nz = standardized != 0
        if nz.sum() > 0:
            _extend(row_indices[nz], col_offsets[nz], standardized[nz])

    # ── Assemble sparse matrix ────────────────────────────────
    rows = rows[:ptr]
    cols = cols[:ptr]
    vals = vals[:ptr]

    X = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(n, info.n_cols),
        dtype=dtype,
    ).tocsr()

    nnz_per_row = X.nnz / max(n, 1)
    mem_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    logger.info(
        "Sparse design matrix: %d x %d, nnz=%d (%.1f/row), %.1f MB",
        X.shape[0], X.shape[1], X.nnz, nnz_per_row, mem_mb,
    )
    return X


def get_column_names(info: DesignInfo) -> List[str]:
    """Return human-readable column names matching the design matrix columns."""
    names = []

    # Block 0: intercept
    names.append("intercept")

    # Block 1: continuous
    for col in info.continuous_cols:
        names.append(f"cont__{col}")

    # Block 2: tract FEs
    for tract in info.tract_categories:
        names.append(f"tract_fe__{tract}")

    # Block 3: tract x feature interactions
    for feat in info.tract_interaction_features:
        for tract in info.tract_categories:
            names.append(f"tract_x_{feat}__{tract}")

    assert len(names) == info.n_cols, (
        f"Column name count {len(names)} != info.n_cols {info.n_cols}"
    )
    return names
