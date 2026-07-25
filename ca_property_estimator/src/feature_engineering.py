"""
Feature engineering for the CA Property Estimator.

Transforms raw parcel data into model-ready features, leveraging
Proposition 13 assessment logic and property characteristics.

Supports both residential and non-residential property types with
type-aware feature creation and validation.
"""

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import classify_property_type
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# IMPROVEMENT MODEL FEATURES
# Used to predict building/improvement value after land value is
# estimated via geographic comparables.
# ══════════════════════════════════════════════════════════════════════

# ── Building characteristics (raw from Snowflake) ────────────────────
IMPROVEMENT_NUMERIC = [
    "BUILDING_SQFT",
    "LIVING_SQFT",
    "ASSR_SQFT_NUM",
    "TOTAL_ROOMS",
    "BEDROOMS",
    "TOTAL_BATHS_CALCULATED",
    "PARTIAL_BATHS",
    "STORIES_NUM",
    "PARKING_SPACES",
    "UNITS_NUMBER",
    "YR_BLT_NUM",
    "YR_BLT_EFFECT_NUM",
]

# ── Engineered building features ─────────────────────────────────────
IMPROVEMENT_ENGINEERED = [
    "PROPERTY_AGE",
    "EFFECTIVE_AGE",
    "BATH_BED_RATIO",
    "ROOMS_PER_SQFT",
    "LIVING_TO_BLDG_RATIO",    # finished vs gross sqft
]

# ── Building categorical features ────────────────────────────────────
IMPROVEMENT_CATEGORICAL = [
    "USE_CODE_STD_DESC_LPS",
    "BUILDING_TYPE_SIMPLIFIED",
    "STYLE_DESC",
    "CONSTRUCTION_CODE_DESC",
    "EXTERIOR_WALL_DESC",
    "ROOF_COVER_DESC",
    "HEATING_DESC",
    "AIR_CONDITIONING_TYPE_DESC",
    "FIREPLACE_DESC",
    "POOL_CODE_DESC",
    "GARAGE_CODE_DESC",
    "PARKING_TYPE",
    "FLOOR_TYPE",
    "BASEMENT_FINISH_DESC",
]

# ══════════════════════════════════════════════════════════════════════
# CONTEXT / LOCATION FEATURES
# Not used directly in improvement model, but available for
# diagnostics and the land comps model.
# ══════════════════════════════════════════════════════════════════════
LOCATION_NUMERIC = [
    "LATITUDE_NUM",
    "LONGITUDE_NUM",
    "LOT_SIZE_AREA",
    "LOT_WIDTH_NUM",
    "LOT_DEPTH_NUM",
    "SHAPE_AREA_NUM",
    "LOCAL_TAX_RATE",
]

LOCATION_CATEGORICAL = [
    "COUNTYNAME",
    "OWNER_OCCUPIED",
    "CA_HOME_OWNERS_EXEMPT",
    "H3_RES5",
    "H3_RES6",
    "H3_RES7",
    "H3_RES8",
    "H3_RES9",
]

# All categorical features (union for cleaning purposes)
ALL_CATEGORICAL = (
    IMPROVEMENT_CATEGORICAL + LOCATION_CATEGORICAL
)


def safe_numeric(df: pd.DataFrame, col: str, new_col: str = None) -> pd.Series:
    """Convert a VARCHAR column to numeric, coercing errors to NaN."""
    if new_col is None:
        new_col = col
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, name=new_col)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features from the raw parcel DataFrame.

    Steps:
      1. Parse string columns to proper numeric types
      2. Build Prop-13-aware derived features
      3. Build physical-property derived features
      4. Clean categorical columns

    Returns a new DataFrame with original + engineered columns.
    """
    df = df.copy()
    logger.info("Starting feature engineering on %d rows", len(df))

    # ── 1. Type conversions ──────────────────────────────────────────
    df["LATITUDE_NUM"] = safe_numeric(df, "LATITUDE")
    df["LONGITUDE_NUM"] = safe_numeric(df, "LONGITUDE")
    df["STORIES_NUM"] = safe_numeric(df, "STORIES_NUMBER")
    df["YR_BLT_NUM"] = safe_numeric(df, "YR_BLT")
    df["YR_BLT_EFFECT_NUM"] = safe_numeric(df, "YR_BLT_EFFECT")
    df["ASSR_SQFT_NUM"] = safe_numeric(df, "ASSR_SQFT")
    df["TAX_AMOUNT_NUM"] = safe_numeric(df, "TAX_AMOUNT")
    df["LOT_WIDTH_NUM"] = safe_numeric(df, "LOT_WIDTH")
    df["LOT_DEPTH_NUM"] = safe_numeric(df, "LOT_DEPTH")
    df["SHAPE_AREA_NUM"] = safe_numeric(df, "SHAPE_AREA")

    # Ensure core numeric columns are actually numeric
    for col in ["VAL_ASSD_LAND", "VAL_ASSD_IMPRV", "VAL_ASSD",
                "AVM_VALUE", "BUILDING_SQFT", "LIVING_SQFT",
                "LOT_SIZE_AREA", "TOTAL_ROOMS", "BEDROOMS",
                "TOTAL_BATHS_CALCULATED", "PARTIAL_BATHS",
                "PARKING_SPACES", "UNITS_NUMBER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse prior sale values
    df["PRIOR_SALE_PRICE"] = safe_numeric(df, "PRIOR_SALE_VAL_TRANSFER")
    df["PRIOR_SALE_DATE"] = pd.to_datetime(
        df.get("PRIOR_SALE_DATE_TRANSFER"), errors="coerce", utc=True
    )

    current_year = pd.Timestamp.now(tz="UTC").year

    # ── 2. Prop 13 derived features ──────────────────────────────────
    # Property age
    df["PROPERTY_AGE"] = current_year - df["YR_BLT_NUM"]
    df.loc[df["PROPERTY_AGE"] < 0, "PROPERTY_AGE"] = np.nan
    df.loc[df["PROPERTY_AGE"] > 300, "PROPERTY_AGE"] = np.nan

    # Effective age (from remodel year)
    df["EFFECTIVE_AGE"] = current_year - df["YR_BLT_EFFECT_NUM"]
    df.loc[df["EFFECTIVE_AGE"] < 0, "EFFECTIVE_AGE"] = np.nan

    # Years since last sale
    if "SALE_DATE_PARSED" in df.columns:
        df["YEARS_SINCE_SALE"] = (
            (pd.Timestamp.now(tz="UTC") - df["SALE_DATE_PARSED"]).dt.days / 365.25
        )
        df.loc[df["YEARS_SINCE_SALE"] < 0, "YEARS_SINCE_SALE"] = np.nan
    else:
        df["YEARS_SINCE_SALE"] = np.nan

    # Assessment value ratios — land vs improvement split
    # (both Prop 13 constrained, but the ratio reveals land-heavy vs improvement-heavy)
    total_assd = df["VAL_ASSD"].replace(0, np.nan)
    df["ASSD_LAND_RATIO"] = df["VAL_ASSD_LAND"] / total_assd
    df["ASSD_IMPRV_RATIO"] = df["VAL_ASSD_IMPRV"] / total_assd

    # Prop 13 drift: how far has assessed value grown from the sale basis?
    # For recently sold properties this should be ~1.0; for long-held ~1.02^years
    sale_price = df["SALE_PRICE"] if "SALE_PRICE" in df.columns else pd.Series(np.nan, index=df.index)
    df["ASSD_TO_SALE_RATIO"] = df["VAL_ASSD"] / sale_price.replace(0, np.nan)
    df["ASSD_TO_SALE_RATIO"] = df["ASSD_TO_SALE_RATIO"].clip(0, 10)

    # Expected Prop 13 growth: 1.02^years since sale
    # Deviation from this reveals reassessments, additions, or data issues
    if "YEARS_SINCE_SALE" in df.columns:
        df["EXPECTED_PROP13_GROWTH"] = 1.02 ** df["YEARS_SINCE_SALE"]
        df["PROP13_DEVIATION"] = df["ASSD_TO_SALE_RATIO"] / df["EXPECTED_PROP13_GROWTH"].replace(0, np.nan)
        df["PROP13_DEVIATION"] = df["PROP13_DEVIATION"].clip(0, 5)
    else:
        df["EXPECTED_PROP13_GROWTH"] = np.nan
        df["PROP13_DEVIATION"] = np.nan

    # Prior sale annual appreciation
    if "SALE_DATE_PARSED" in df.columns:
        years_between = (
            (df["SALE_DATE_PARSED"] - df["PRIOR_SALE_DATE"]).dt.days / 365.25
        )
        years_between = years_between.replace(0, np.nan)
        price_ratio = df["SALE_PRICE"] / df["PRIOR_SALE_PRICE"].replace(0, np.nan)
        df["PRIOR_ANNUAL_APPRECIATION"] = np.where(
            (years_between > 0) & (price_ratio > 0),
            (price_ratio ** (1.0 / years_between)) - 1.0,
            np.nan,
        )
        # Cap extreme values
        df["PRIOR_ANNUAL_APPRECIATION"] = df["PRIOR_ANNUAL_APPRECIATION"].clip(-0.5, 2.0)
    else:
        df["PRIOR_ANNUAL_APPRECIATION"] = np.nan

    # Local tax rate (TAX_AMOUNT / VAL_ASSD)
    # Under Prop 13 this is ~1% + local bonds, so it's a proxy for
    # LOCAL TAX BURDEN / DISTRICT — not a property-specific signal
    # but still useful as a location feature
    df["LOCAL_TAX_RATE"] = df["TAX_AMOUNT_NUM"] / df["VAL_ASSD"].replace(0, np.nan)
    df["LOCAL_TAX_RATE"] = df["LOCAL_TAX_RATE"].clip(0, 0.05)

    # Tax amount per square foot — combines tax district with density
    df["TAX_PER_SQFT"] = df["TAX_AMOUNT_NUM"] / df["BUILDING_SQFT"].replace(0, np.nan)
    df["TAX_PER_SQFT"] = df["TAX_PER_SQFT"].clip(0, 100)

    # Improvement to land ratio
    df["IMPRV_TO_LAND_RATIO"] = (
        df["VAL_ASSD_IMPRV"] / df["VAL_ASSD_LAND"].replace(0, np.nan)
    )
    df["IMPRV_TO_LAND_RATIO"] = df["IMPRV_TO_LAND_RATIO"].clip(0, 100)

    # ── 3. Physical property derived features ────────────────────────
    bldg_sqft = df["BUILDING_SQFT"].replace(0, np.nan)
    living_sqft = df["LIVING_SQFT"].replace(0, np.nan)

    df["ASSD_PER_SQFT"] = df["VAL_ASSD"] / bldg_sqft
    # Use AVM (independent automated valuation) instead of VAL_MARKET
    # since VAL_MARKET is Prop 13 constrained, same as VAL_ASSD
    df["AVM_PER_SQFT"] = df["AVM_VALUE"].replace(0, np.nan) / bldg_sqft

    df["PRICE_PER_SQFT_LIVING"] = np.where(
        df["IS_TRAINING"] if "IS_TRAINING" in df.columns else False,
        df.get("SALE_PRICE", np.nan) / living_sqft,
        np.nan,
    )

    df["LOT_TO_BLDG_RATIO"] = df["LOT_SIZE_AREA"] / bldg_sqft
    df["LOT_TO_BLDG_RATIO"] = df["LOT_TO_BLDG_RATIO"].clip(0, 1000)

    beds = df["BEDROOMS"].replace(0, np.nan)
    df["BATH_BED_RATIO"] = df["TOTAL_BATHS_CALCULATED"] / beds

    rooms = df["TOTAL_ROOMS"].replace(0, np.nan)
    df["ROOMS_PER_SQFT"] = rooms / living_sqft

    # ── 4. Building-specific derived features ────────────────────────
    df["LIVING_TO_BLDG_RATIO"] = living_sqft / bldg_sqft
    df["LIVING_TO_BLDG_RATIO"] = df["LIVING_TO_BLDG_RATIO"].clip(0, 1.5)

    # ── 5. Clean categoricals ────────────────────────────────────────
    for col in ALL_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
            # Collapse rare categories (< 0.5% of data) into OTHER
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < 0.005].index
            df.loc[df[col].isin(rare), col] = "OTHER"

    # ── 6. Property type classification ──────────────────────────────
    # Classify each parcel into a model group using USE_CODE_STD_LPS.
    # This drives feature routing (e.g., BEDROOMS is meaningless for
    # commercial) and model selection in the hierarchical architecture.
    if "USE_CODE_STD_LPS" in df.columns:
        df["PROPERTY_TYPE_GROUP"] = df["USE_CODE_STD_LPS"].apply(
            classify_property_type
        )
        type_counts = df["PROPERTY_TYPE_GROUP"].value_counts()
        logger.info("Property type classification:")
        for ptype, cnt in type_counts.items():
            logger.info("  %s: %d (%.1f%%)", ptype, cnt, 100 * cnt / len(df))
    else:
        df["PROPERTY_TYPE_GROUP"] = "unknown"

    # ── 7. Non-residential features ──────────────────────────────────
    # LOT_TO_BLDG_RATIO is critical for commercial/industrial where
    # land-to-building proportions vary dramatically
    # (already computed as LOT_TO_BLDG_RATIO above in section 3)

    # STORIES_NUM already parsed in section 1

    # BLDG_CLASS — building class for commercial/industrial (44-55% populated)
    # Keep as-is for later dummy encoding in sparse_features

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def get_feature_lists(df: pd.DataFrame, stage: str = "improvement") -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical feature columns for a given stage.

    Stages:
        'improvement' — building characteristics for improvement value model
        'full'        — all features (improvement + location context)
    """
    if stage == "improvement":
        num_raw = IMPROVEMENT_NUMERIC + IMPROVEMENT_ENGINEERED
        cat_raw = IMPROVEMENT_CATEGORICAL
    elif stage == "full":
        num_raw = IMPROVEMENT_NUMERIC + IMPROVEMENT_ENGINEERED + LOCATION_NUMERIC
        cat_raw = IMPROVEMENT_CATEGORICAL + LOCATION_CATEGORICAL
    else:
        raise ValueError(f"Unknown stage: {stage}")

    num_cols = [c for c in num_raw if c in df.columns]
    cat_cols = [c for c in cat_raw if c in df.columns]
    # Deduplicate while preserving order
    num_cols = list(dict.fromkeys(num_cols))
    cat_cols = list(dict.fromkeys(cat_cols))
    return num_cols, cat_cols


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    max_cat_cardinality: int = 100,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - Imputes + scales numeric features
      - One-hot encodes categorical features (with cardinality cap)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=max_cat_cardinality,
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = "IMPROVEMENT_VALUE",
    log_transform: bool = True,
    stage: str = "improvement",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepare training data and full dataset for the improvement model.

    In the comps-based approach:
      - Land value is estimated geographically (see comparables.py)
      - This prepares data for the IMPROVEMENT value model
      - target_col should be 'IMPROVEMENT_VALUE' (sale_price - est_land)

    Returns:
        X_train_df: DataFrame of features for training rows
        X_all_df:   DataFrame of features for all rows
        y_train:    Target series for training rows
        preprocessor: Fitted ColumnTransformer
    """
    num_cols, cat_cols = get_feature_lists(df, stage=stage)
    logger.info("Stage '%s': %d numeric, %d categorical features",
                stage, len(num_cols), len(cat_cols))

    train_mask = (
        (df["IS_TRAINING"] == True)
        & df[target_col].notna()
        & (df[target_col] > 0)
    )
    X_train_df = df.loc[train_mask, num_cols + cat_cols].copy()
    X_all_df = df[num_cols + cat_cols].copy()

    y_train = df.loc[train_mask, target_col].copy()
    if log_transform:
        y_train = np.log1p(y_train)

    logger.info("Training on %d properties with valid %s",
                len(y_train), target_col)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train_df)

    return X_train_df, X_all_df, y_train, preprocessor
