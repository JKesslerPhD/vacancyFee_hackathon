"""
Integration test for the sparse GLM pipeline v2.

Uses synthetic data to verify:
  1. DesignInfo fitting (with HAS_AC and construction dummies as continuous)
  2. Sparse design matrix construction (with global intercept)
  3. Matrix dimensions and sparsity
  4. OLS fitting via direct sparse solve
  5. Prediction with YEARS_SINCE_SALE=0
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from src.sparse_features import (
    DesignInfo,
    fit_design_info,
    build_design_matrix,
    get_column_names,
)


def make_synthetic_data(n=5000, n_tracts=50, seed=42):
    """Create synthetic CA property data for testing."""
    rng = np.random.default_rng(seed)

    tracts = [f"06{i:04d}" for i in range(n_tracts)]
    construction_types = ["WOOD FRAME", "MASONRY", "STEEL", "CONCRETE", "OTHER"]
    ac_types = ["CENTRAL", "NONE", "WINDOW UNIT", "UNKNOWN"]

    df = pd.DataFrame({
        "BUILDING_SQFT": rng.uniform(800, 5000, n),
        "LOT_SIZE_AREA": rng.uniform(2000, 20000, n),
        "BEDROOMS": rng.integers(1, 6, n).astype(float),
        "TOTAL_BATHS_CALCULATED": rng.uniform(1, 4, n),
        "PROPERTY_AGE": rng.uniform(0, 100, n),
        "YEARS_SINCE_SALE": rng.uniform(0, 5, n),
        "CENSUS_TRACT": rng.choice(tracts, n),
        "CONSTRUCTION_CODE_DESC": rng.choice(construction_types, n),
        "AIR_CONDITIONING_TYPE_DESC": rng.choice(ac_types, n),
        "LATITUDE": rng.uniform(32, 42, n),
        "LONGITUDE": rng.uniform(-124, -114, n),
    })

    # Simulate sale prices based on features
    tract_values = {t: rng.uniform(11, 14) for t in tracts}
    log_price = np.array([tract_values[t] for t in df["CENSUS_TRACT"]])
    log_price += 0.0003 * df["BUILDING_SQFT"].values
    log_price += 0.00005 * df["LOT_SIZE_AREA"].values
    log_price += 0.02 * df["BEDROOMS"].values
    log_price += 0.05 * df["TOTAL_BATHS_CALCULATED"].values
    log_price -= 0.002 * df["PROPERTY_AGE"].values
    log_price += rng.normal(0, 0.1, n)
    df["SALE_PRICE"] = np.expm1(log_price)

    return df


def _fit_info(df):
    """Helper to fit DesignInfo with standard args."""
    return fit_design_info(
        df,
        continuous_cols=["BUILDING_SQFT", "LOT_SIZE_AREA",
                         "BEDROOMS", "TOTAL_BATHS_CALCULATED",
                         "PROPERTY_AGE", "YEARS_SINCE_SALE"],
        tract_col="CENSUS_TRACT",
        tract_interaction_features=["BUILDING_SQFT", "LOT_SIZE_AREA"],
        construction_col="CONSTRUCTION_CODE_DESC",
        ac_col="AIR_CONDITIONING_TYPE_DESC",
        min_tract_sales=10,
    )


class TestDesignInfo:
    """Test the fit_design_info function."""

    def setup_method(self):
        self.df = make_synthetic_data()

    def test_fit_creates_valid_info(self):
        info = _fit_info(self.df)
        assert info.n_cols > 0
        assert len(info.tract_categories) > 0
        assert info.pooled_tract_label in info.tract_categories
        # 6 original + HAS_AC + construction dummies
        assert len(info.continuous_cols) >= 7  # at least +1 for HAS_AC
        assert "HAS_AC" in info.continuous_cols
        # Should have intercept block
        assert "intercept" in info.block_ranges

    def test_column_names_match_ncols(self):
        info = _fit_info(self.df)
        names = get_column_names(info)
        assert len(names) == info.n_cols


class TestDesignMatrix:
    """Test sparse design matrix construction."""

    def setup_method(self):
        self.df = make_synthetic_data()
        self.info = _fit_info(self.df)

    def test_matrix_shape(self):
        X = build_design_matrix(self.df, self.info)
        assert X.shape == (len(self.df), self.info.n_cols)

    def test_matrix_is_sparse(self):
        X = build_design_matrix(self.df, self.info)
        density = X.nnz / (X.shape[0] * X.shape[1])
        assert density < 0.15, f"Matrix too dense: {density:.3f}"

    def test_years_since_sale_zeroed(self):
        X_train = build_design_matrix(
            self.df, self.info, set_years_since_sale_zero=False)
        X_pred = build_design_matrix(
            self.df, self.info, set_years_since_sale_zero=True)

        yss_idx = self.info.continuous_cols.index("YEARS_SINCE_SALE")
        col_idx = self.info.block_ranges["continuous"][0] + yss_idx

        train_col = X_train[:, col_idx].toarray().ravel()
        pred_col = X_pred[:, col_idx].toarray().ravel()

        assert np.any(train_col != 0)
        assert np.std(pred_col) < 0.01

    def test_every_row_has_tract(self):
        """Every row should have exactly one tract FE = 1."""
        X = build_design_matrix(self.df, self.info)
        fe_start, fe_end = self.info.block_ranges["tract_fe"]
        tract_block = X[:, fe_start:fe_end].toarray()
        row_sums = tract_block.sum(axis=1)
        np.testing.assert_array_equal(row_sums, 1.0)

    def test_intercept_column_all_ones(self):
        """The intercept column should be 1.0 for every row."""
        X = build_design_matrix(self.df, self.info)
        intercept_col = X[:, 0].toarray().ravel()
        np.testing.assert_array_equal(intercept_col, 1.0)


class TestEndToEnd:
    """Test full fit -> build -> predict cycle."""

    def test_ols_fit_and_predict(self):
        """Fit OLS on synthetic data and verify predictions are reasonable."""
        from src.pipeline import fit_glm

        df = make_synthetic_data(n=2000, n_tracts=20)
        info = _fit_info(df)

        X = build_design_matrix(df, info, set_years_since_sale_zero=False)
        y = np.log1p(df["SALE_PRICE"].values).astype(np.float32)
        column_names = get_column_names(info)

        # Fit OLS via sparse direct solve
        results = fit_glm(X, y, column_names)

        # R² should be high on synthetic data
        assert results.rsquared > 0.8, (
            f"R² too low on synthetic data: {results.rsquared:.3f}"
        )

        # Predict current values (YEARS_SINCE_SALE=0)
        X_pred = build_design_matrix(
            df, info, set_years_since_sale_zero=True)
        y_pred_log = X_pred.dot(results.params)
        est_values = np.expm1(y_pred_log)

        # Values should be positive and reasonable
        assert np.all(est_values > 0)
        assert np.median(est_values) > 10_000
        assert np.median(est_values) < 50_000_000

        print(f"\n  OLS R² = {results.rsquared:.4f}")
        print(f"  RMSE (log) = {results.rmse:.4f}")
        print(f"  Median estimate: ${np.median(est_values):,.0f}")
        print(f"  Parameters: {info.n_cols}")
        print(f"  Obs/param: {results.n_obs / results.n_params:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])