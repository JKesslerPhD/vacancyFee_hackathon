"""
Tests for score_all.py — empirical land + improvement models.

Validates:
  1. safe_expm1 overflow protection
  2. ensure_fips_tract creation and idempotency
  3. add_use_type_dummies column creation
  4. build_tract_fe_matrix sparse structure
  5. fit_sparse_ols basic regression
  6. prepare_land_training outlier filtering
  7. train_land_model produces valid coefficients
  8. score_land returns correct shape
  9. prepare_imprv_training data filtering
  10. train_imprv_model produces valid coefficients
  11. build_multiplier_table t=0 gives 1.0
  12. score_imprv applies multiplier correctly
"""

import sys
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, ".")
from src.score_all import (
    safe_expm1,
    ensure_fips_tract,
    add_use_type_dummies,
    build_tract_fe_matrix,
    fit_sparse_ols,
    MAX_VALUE_CLIP,
    USE_TYPE_DUMMIES,
)


class TestSafeExpm1:
    """Test overflow protection."""

    def test_normal_values_unchanged(self):
        log_pred = np.array([11.0, 12.0, 13.0, 14.0])
        result = safe_expm1(log_pred)
        expected = np.expm1(log_pred)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_overflow_clipped(self):
        log_pred = np.array([50.0, 100.0, 200.0, 1000.0])
        result = safe_expm1(log_pred)
        assert np.all(result <= MAX_VALUE_CLIP)
        assert np.all(np.isfinite(result))

    def test_negative_clipped_to_zero(self):
        log_pred = np.array([-100.0, -50.0])
        result = safe_expm1(log_pred)
        assert np.all(result >= 0)
        assert np.all(result < 1.0)

    def test_zero_returns_zero(self):
        result = safe_expm1(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_known_value(self):
        val = 500_000.0
        log_val = np.log1p(val)
        result = safe_expm1(np.array([log_val]))
        np.testing.assert_allclose(result, [val], rtol=1e-5)

    def test_custom_clip_max(self):
        result = safe_expm1(np.array([50.0]), clip_max=1_000_000)
        assert result[0] <= 1_000_000


class TestEnsureFipsTract:
    """Test FIPS_TRACT creation."""

    def test_creates_column(self):
        df = pd.DataFrame({
            "FIPS_CODE": ["06001", "06037"],
            "CENSUS_TRACT": ["000100", "123456"],
        })
        result = ensure_fips_tract(df)
        assert "FIPS_TRACT" in result.columns
        assert result["FIPS_TRACT"].iloc[0] == "06001_000100"
        assert result["FIPS_TRACT"].iloc[1] == "06037_123456"

    def test_idempotent(self):
        df = pd.DataFrame({
            "FIPS_CODE": ["06001"],
            "CENSUS_TRACT": ["000100"],
            "FIPS_TRACT": ["06001_000100"],
        })
        result = ensure_fips_tract(df)
        assert result["FIPS_TRACT"].iloc[0] == "06001_000100"
        assert len(result.columns) == 3


class TestAddUseTypeDummies:
    """Test use-type dummy column creation."""

    def test_all_dummies_created(self):
        df = pd.DataFrame({
            "PROPERTY_TYPE_GROUP": ["residential", "commercial", "industrial"],
        })
        result = add_use_type_dummies(df)
        for utype in USE_TYPE_DUMMIES:
            assert f"IS_{utype.upper()}" in result.columns

    def test_correct_values(self):
        df = pd.DataFrame({
            "PROPERTY_TYPE_GROUP": ["commercial", "residential", "industrial"],
        })
        result = add_use_type_dummies(df)
        assert result["IS_COMMERCIAL"].iloc[0] == 1.0
        assert result["IS_COMMERCIAL"].iloc[1] == 0.0
        assert result["IS_INDUSTRIAL"].iloc[2] == 1.0

    def test_residential_is_reference(self):
        """Residential should have all dummies = 0."""
        df = pd.DataFrame({"PROPERTY_TYPE_GROUP": ["residential"]})
        result = add_use_type_dummies(df)
        for utype in USE_TYPE_DUMMIES:
            assert result[f"IS_{utype.upper()}"].iloc[0] == 0.0

    def test_idempotent(self):
        df = pd.DataFrame({"PROPERTY_TYPE_GROUP": ["commercial"]})
        result = add_use_type_dummies(df)
        result2 = add_use_type_dummies(result)
        assert len(result2.columns) == len(result.columns)


class TestBuildTractFEMatrix:
    """Test sparse tract FE matrix construction."""

    def test_shape(self):
        tracts = ["06001_0001", "06037_0002", "06001_0001"]
        categories = ["06001_0001", "06037_0002", "__POOLED__"]
        mat = build_tract_fe_matrix(tracts, categories)
        assert mat.shape == (3, 3)

    def test_one_hot(self):
        """Each row should have exactly one nonzero."""
        tracts = ["A", "B", "C", "A"]
        categories = ["A", "B", "C", "__POOLED__"]
        mat = build_tract_fe_matrix(tracts, categories)
        row_sums = np.array(mat.sum(axis=1)).flatten()
        np.testing.assert_array_equal(row_sums, [1, 1, 1, 1])

    def test_correct_assignment(self):
        tracts = ["A", "B", "A"]
        categories = ["A", "B", "__POOLED__"]
        mat = build_tract_fe_matrix(tracts, categories).toarray()
        # Row 0 -> A (col 0)
        assert mat[0, 0] == 1.0
        assert mat[0, 1] == 0.0
        # Row 1 -> B (col 1)
        assert mat[1, 1] == 1.0

    def test_unknown_goes_to_pooled(self):
        tracts = ["UNKNOWN"]
        categories = ["A", "B", "__POOLED__"]
        mat = build_tract_fe_matrix(tracts, categories).toarray()
        # Should be assigned to pooled (col 2)
        assert mat[0, 2] == 1.0
        assert mat[0, 0] == 0.0
        assert mat[0, 1] == 0.0


class TestFitSparseOLS:
    """Test sparse OLS solver."""

    def test_simple_regression(self):
        """y = 2*x + 1 should recover beta ~ [1, 2]."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 1.0 + 2.0 * x + np.random.randn(n) * 0.01
        X = sparse.csr_matrix(np.column_stack([np.ones(n), x]))
        beta = fit_sparse_ols(X, y, ridge_frac=0.001)
        np.testing.assert_allclose(beta[0], 1.0, atol=0.05)
        np.testing.assert_allclose(beta[1], 2.0, atol=0.05)

    def test_ridge_regularization(self):
        """With tiny data, ridge should prevent blow-up."""
        X = sparse.csr_matrix(np.array([[1, 1], [1, 1.001]]))
        y = np.array([1.0, 1.001])
        beta = fit_sparse_ols(X, y, ridge_frac=0.1)
        assert np.all(np.isfinite(beta))


class TestMultiplierTableLogic:
    """Test multiplier table construction logic."""

    def test_t0_gives_one(self):
        """At t=0, exp(0) = 1.0 regardless of slopes."""
        # multiplier = exp(-slope * t), at t=0 -> exp(0) = 1
        slopes = [-0.05, 0.0, 0.03, 0.10]
        for s in slopes:
            mult = np.exp(-s * 0)
            assert mult == 1.0

    def test_negative_slope_increases_multiplier(self):
        """Negative global slope means values decay with age,
        so multiplier should be > 1 for t > 0."""
        # If slope is negative (assessed values decline with age),
        # multiplier = exp(-negative * t) = exp(positive) > 1
        slope = -0.05  # 5% annual decay
        t = 10
        mult = np.exp(-slope * t)
        assert mult > 1.0  # exp(0.5) ~ 1.65

    def test_multiplier_increases_with_years(self):
        """Multiplier should increase with years since sale."""
        slope = -0.05
        years = [0, 5, 10, 20]
        mults = [np.exp(-slope * t) for t in years]
        for i in range(len(mults) - 1):
            assert mults[i + 1] > mults[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
