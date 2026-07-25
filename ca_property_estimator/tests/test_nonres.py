"""
Tests for non-residential expansion.

Validates:
  1. Property type classification from USE_CODE_STD_LPS
  2. Model routing by property type
  3. Tract FE extraction from fitted model
  4. Tract FE application to non-res DataFrame
  5. Feature engineering with property type groups
  6. prepare_training_df with allowed_property_types filter
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from src.config import (
    classify_property_type,
    get_model_for_property_type,
    PROPERTY_TYPE_RANGES,
    MODEL_ROUTING,
    GLMConfig,
)
from src.feature_engineering import engineer_features
from src.pipeline import extract_tract_fe, apply_tract_fe_to_df


class TestPropertyTypeClassification:
    """Test USE_CODE_STD_LPS -> property type mapping."""

    def test_residential_codes(self):
        assert classify_property_type("1001") == "residential"
        assert classify_property_type("1099") == "residential"
        assert classify_property_type(1001) == "residential"

    def test_condo_codes(self):
        assert classify_property_type("1100") == "condo"
        assert classify_property_type("1104") == "condo"
        assert classify_property_type("1199") == "condo"

    def test_multifamily_codes(self):
        assert classify_property_type("1200") == "multifamily"
        assert classify_property_type("1300") == "multifamily"
        assert classify_property_type("1399") == "multifamily"

    def test_commercial_codes(self):
        assert classify_property_type("2001") == "commercial"
        assert classify_property_type("2044") == "commercial"
        assert classify_property_type("2999") == "commercial"

    def test_industrial_codes(self):
        assert classify_property_type("3003") == "industrial"
        assert classify_property_type("3999") == "industrial"

    def test_agricultural_codes(self):
        assert classify_property_type("4000") == "agricultural"
        assert classify_property_type("4999") == "agricultural"

    def test_vacant_land_codes(self):
        assert classify_property_type("5000") == "vacant_land"
        assert classify_property_type("5999") == "vacant_land"

    def test_institutional_codes(self):
        assert classify_property_type("6000") == "institutional"

    def test_recreation_codes(self):
        assert classify_property_type("7005") == "recreation"

    def test_unknown_codes(self):
        assert classify_property_type("9999") == "unknown"
        assert classify_property_type("0010") == "unknown"
        assert classify_property_type(None) == "unknown"
        assert classify_property_type("") == "unknown"
        assert classify_property_type("bad") == "unknown"

    def test_all_ranges_covered(self):
        """Every defined range should classify correctly."""
        for group_name, (lo, hi) in PROPERTY_TYPE_RANGES.items():
            assert classify_property_type(lo) == group_name
            assert classify_property_type(hi) == group_name
            mid = (lo + hi) // 2
            assert classify_property_type(mid) == group_name


class TestModelRouting:
    """Test property type -> model name routing."""

    def test_residential_routes(self):
        assert get_model_for_property_type("residential") == "residential_v4"
        assert get_model_for_property_type("condo") == "residential_v4"

    def test_nonres_routes(self):
        assert get_model_for_property_type("commercial") == "nonres_glm"
        assert get_model_for_property_type("industrial") == "nonres_glm"
        assert get_model_for_property_type("multifamily") == "nonres_glm"

    def test_fallback_routes(self):
        assert get_model_for_property_type("agricultural") == "land_fallback"
        assert get_model_for_property_type("institutional") == "land_fallback"

    def test_unknown_defaults_to_fallback(self):
        assert get_model_for_property_type("unknown") == "land_fallback"
        assert get_model_for_property_type("nonexistent") == "land_fallback"

    def test_all_types_have_routes(self):
        for group_name in PROPERTY_TYPE_RANGES:
            model = get_model_for_property_type(group_name)
            assert model in ("residential_v4", "nonres_glm",
                             "land_model", "land_fallback")


class TestTractFEExtraction:
    """Test extracting and applying tract FE from a fitted model."""

    def _fit_model(self):
        """Fit a small model and return results + design_info."""
        from tests.test_sparse_glm import make_synthetic_data, _fit_info
        from src.sparse_features import build_design_matrix, get_column_names
        from src.pipeline import fit_glm

        df = make_synthetic_data(n=2000, n_tracts=20)
        info = _fit_info(df)
        X = build_design_matrix(df, info, set_years_since_sale_zero=False)
        y = np.log1p(df["SALE_PRICE"].values).astype(np.float32)
        column_names = get_column_names(info)
        results = fit_glm(X, y, column_names)
        return results, info, df

    def test_extract_returns_dict(self):
        results, info, _ = self._fit_model()
        fe_dict = extract_tract_fe(results, info)
        assert isinstance(fe_dict, dict)
        assert len(fe_dict) == len(info.tract_categories)

    def test_extract_includes_pooled(self):
        results, info, _ = self._fit_model()
        fe_dict = extract_tract_fe(results, info)
        assert "__POOLED__" in fe_dict

    def test_extract_values_are_floats(self):
        results, info, _ = self._fit_model()
        fe_dict = extract_tract_fe(results, info)
        for v in fe_dict.values():
            assert isinstance(v, float)

    def test_apply_to_df(self):
        results, info, df = self._fit_model()
        fe_dict = extract_tract_fe(results, info)
        df_out = apply_tract_fe_to_df(df, fe_dict, tract_col="CENSUS_TRACT")
        assert "RES_TRACT_FE" in df_out.columns
        assert df_out["RES_TRACT_FE"].notna().all()
        # Values should be non-trivial (not all the same)
        assert df_out["RES_TRACT_FE"].std() > 0.01

    def test_apply_unknown_tract_gets_pooled(self):
        results, info, _ = self._fit_model()
        fe_dict = extract_tract_fe(results, info)
        pooled_val = fe_dict["__POOLED__"]

        df_new = pd.DataFrame({
            "CENSUS_TRACT": ["UNKNOWN_TRACT_999", "ANOTHER_UNKNOWN"]
        })
        df_out = apply_tract_fe_to_df(df_new, fe_dict, tract_col="CENSUS_TRACT")
        np.testing.assert_array_almost_equal(
            df_out["RES_TRACT_FE"].values, [pooled_val, pooled_val]
        )


class TestFeatureEngineeringPropertyType:
    """Test that engineer_features adds PROPERTY_TYPE_GROUP."""

    def test_property_type_added(self):
        df = pd.DataFrame({
            "USE_CODE_STD_LPS": ["1001", "1104", "2001", "3003", "7005"],
            "BUILDING_SQFT": [1500, 1200, 5000, 10000, 2000],
            "LIVING_SQFT": [1400, 1100, 4800, 9000, 1800],
            "LOT_SIZE_AREA": [6000, 3000, 10000, 20000, 5000],
            "BEDROOMS": [3, 2, np.nan, np.nan, np.nan],
            "TOTAL_BATHS_CALCULATED": [2, 1.5, np.nan, np.nan, np.nan],
            "TOTAL_ROOMS": [7, 5, np.nan, np.nan, np.nan],
            "YR_BLT": ["1985", "2000", "1970", "1960", "1990"],
            "VAL_ASSD": [500000, 300000, 1000000, 800000, 400000],
            "VAL_ASSD_LAND": [200000, 150000, 600000, 400000, 200000],
            "VAL_ASSD_IMPRV": [300000, 150000, 400000, 400000, 200000],
            "AVM_VALUE": [700000, 500000, np.nan, np.nan, np.nan],
        })
        result = engineer_features(df)
        assert "PROPERTY_TYPE_GROUP" in result.columns
        expected = ["residential", "condo", "commercial",
                    "industrial", "recreation"]
        assert list(result["PROPERTY_TYPE_GROUP"]) == expected

    def test_missing_use_code_gets_unknown(self):
        """Without USE_CODE_STD_LPS, all parcels get 'unknown'."""
        df = pd.DataFrame({
            "BUILDING_SQFT": [1500],
            "LIVING_SQFT": [1400],
            "LOT_SIZE_AREA": [6000],
            "VAL_ASSD": [500000],
            "VAL_ASSD_LAND": [200000],
            "VAL_ASSD_IMPRV": [300000],
            "AVM_VALUE": [700000],
            "BEDROOMS": [3],
            "TOTAL_BATHS_CALCULATED": [2],
            "TOTAL_ROOMS": [7],
        })
        result = engineer_features(df)
        assert result["PROPERTY_TYPE_GROUP"].iloc[0] == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])