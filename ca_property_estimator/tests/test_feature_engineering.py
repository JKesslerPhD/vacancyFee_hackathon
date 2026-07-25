"""
Unit tests for feature engineering module.

Uses synthetic data to validate transformations without
needing a Snowflake connection.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    safe_numeric,
    engineer_features,
    get_feature_lists,
    build_preprocessor,
)


@pytest.fixture
def sample_df():
    """Create a minimal synthetic parcel DataFrame."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "PARCEL_APN": [f"APN{i:06d}" for i in range(n)],
        "COUNTYNAME": np.random.choice(
            ["LOS ANGELES", "SAN FRANCISCO", "SACRAMENTO"], n
        ),
        "LATITUDE": np.random.uniform(32, 42, n).astype(str),
        "LONGITUDE": np.random.uniform(-124, -114, n).astype(str),
        "VAL_ASSD_LAND": np.random.uniform(50000, 500000, n),
        "VAL_ASSD_IMPRV": np.random.uniform(100000, 800000, n),
        "VAL_ASSD": np.random.uniform(200000, 1200000, n),
        "VAL_MRKT_LAND": np.random.uniform(50000, 600000, n),
        "VAL_MRKT_IMPRV": np.random.uniform(100000, 900000, n),

        "AVM_VALUE": np.random.uniform(300000, 1500000, n),
        "BUILDING_SQFT": np.random.uniform(800, 5000, n),
        "LIVING_SQFT": np.random.uniform(700, 4500, n),
        "LOT_SIZE_AREA": np.random.uniform(2000, 20000, n),
        "TOTAL_ROOMS": np.random.randint(3, 12, n).astype(float),
        "BEDROOMS": np.random.randint(1, 6, n).astype(float),
        "TOTAL_BATHS_CALCULATED": np.random.uniform(1, 5, n),
        "PARTIAL_BATHS": np.random.choice([0, 1], n).astype(float),
        "STORIES_NUMBER": np.random.choice(["1", "2", "3"], n),
        "PARKING_SPACES": np.random.choice([0, 1, 2, 3], n).astype(float),
        "UNITS_NUMBER": np.ones(n),
        "YR_BLT": np.random.randint(1920, 2024, n).astype(str),
        "YR_BLT_EFFECT": np.random.randint(1950, 2024, n).astype(str),
        "TAX_AMOUNT": np.random.uniform(2000, 20000, n).astype(str),
        "ASSR_SQFT": np.random.uniform(800, 5000, n).astype(str),
        "LOT_WIDTH": np.random.uniform(30, 100, n).astype(str),
        "LOT_DEPTH": np.random.uniform(80, 200, n).astype(str),
        "SHAPE_AREA": np.random.uniform(2000, 20000, n).astype(str),
        "USE_CODE_STD_DESC_LPS": np.random.choice(
            ["SINGLE FAMILY RESIDENCE", "CONDO", "DUPLEX"], n
        ),
        "BUILDING_TYPE_SIMPLIFIED": np.random.choice(
            ["SFR", "CONDO", "MFR"], n
        ),
        "STYLE_DESC": "RANCH",
        "CONSTRUCTION_CODE_DESC": "WOOD FRAME",
        "EXTERIOR_WALL_DESC": "STUCCO",
        "ROOF_COVER_DESC": "TILE",
        "HEATING_DESC": "CENTRAL",
        "AIR_CONDITIONING_TYPE_DESC": "CENTRAL",
        "FIREPLACE_DESC": np.random.choice(["YES", "NO"], n),
        "POOL_CODE_DESC": np.random.choice(["YES", "NO"], n),
        "GARAGE_CODE_DESC": "ATTACHED",
        "OWNER_OCCUPIED": np.random.choice(["Y", "N"], n),
        "CA_HOME_OWNERS_EXEMPT": np.random.choice(["Y", "N"], n),
        "LOAN_TYPE_1_DESC": "CONVENTIONAL",
        "H3_RES5": "8528a5fffffffff",
        "H3_RES7": "8728a5effffffff",
        "IS_TRAINING": [True] * 150 + [False] * 50,
        "SALE_PRICE": np.concatenate([
            np.random.uniform(300000, 1500000, 150),
            [np.nan] * 50,
        ]),
        "SALE_DATE_PARSED": pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC"),
    })
    # Add prior sale info
    df["PRIOR_SALE_VAL_TRANSFER"] = (
        df["SALE_PRICE"] * np.random.uniform(0.5, 0.9, n)
    ).astype(str)
    df["PRIOR_SALE_DATE_TRANSFER"] = (
        df["SALE_DATE_PARSED"] - pd.DateOffset(years=5)
    ).astype(str)
    return df


def test_safe_numeric():
    df = pd.DataFrame({"A": ["1.5", "bad", "3", None, ""]})
    result = safe_numeric(df, "A")
    assert result.notna().sum() == 2
    assert result.iloc[0] == 1.5


def test_engineer_features_shape(sample_df):
    result = engineer_features(sample_df)
    assert len(result) == len(sample_df)
    assert "PROPERTY_AGE" in result.columns
    assert "ASSD_LAND_RATIO" in result.columns
    assert "IMPRV_TO_LAND_RATIO" in result.columns


def test_engineered_values_reasonable(sample_df):
    result = engineer_features(sample_df)
    assert result["PROPERTY_AGE"].dropna().min() >= 0
    # ASSD_LAND_RATIO can exceed 1.0 when VAL_ASSD_LAND > VAL_ASSD
    # (happens in real data too — assessor inconsistencies)
    assert result["ASSD_LAND_RATIO"].dropna().max() <= 2.0
    assert result["LOCAL_TAX_RATE"].dropna().max() <= 0.05


def test_get_feature_lists_improvement(sample_df):
    result = engineer_features(sample_df)
    num_cols, cat_cols = get_feature_lists(result, stage="improvement")
    assert len(num_cols) > 5
    assert len(cat_cols) > 3
    assert "BUILDING_SQFT" in num_cols
    assert "BUILDING_TYPE_SIMPLIFIED" in cat_cols
    # Location features should NOT be in improvement stage
    assert "COUNTYNAME" not in cat_cols


def test_get_feature_lists_full(sample_df):
    result = engineer_features(sample_df)
    num_cols, cat_cols = get_feature_lists(result, stage="full")
    assert len(num_cols) > 10
    assert "COUNTYNAME" in cat_cols
    assert "LATITUDE_NUM" in num_cols


def test_preprocessor_output(sample_df):
    result = engineer_features(sample_df)
    num_cols, cat_cols = get_feature_lists(result, stage="improvement")
    preprocessor = build_preprocessor(num_cols, cat_cols)
    train_mask = result["IS_TRAINING"] == True
    X_train = result.loc[train_mask, num_cols + cat_cols]
    preprocessor.fit(X_train)
    X_out = preprocessor.transform(X_train)
    assert X_out.shape[0] == train_mask.sum()
    assert X_out.shape[1] > len(num_cols)  # cat encoding adds cols
    assert not np.any(np.isnan(X_out))