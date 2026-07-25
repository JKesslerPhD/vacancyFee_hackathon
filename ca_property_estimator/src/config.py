"""
Configuration loader for the CA Property Estimator.

Reads config.yaml and provides typed access to all settings.
Supports both the legacy YAML-based config and the new GLM pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import yaml


# ═══════════════════════════════════════════════════════════════════
# Existing YAML-based config (preserved for data_loader compatibility)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SnowflakeConfig:
    account: str = ""
    user: str = ""
    password: Optional[str] = None
    authenticator: Optional[str] = None
    warehouse: str = ""
    database: str = "BUILDING_CARBON"
    schema: str = "PARCEL_DATA_2025Q1"
    role: Optional[str] = None


@dataclass
class DataConfig:
    table: str = "PARCELS_CLEAN_2025Q1"
    state_filter: str = "CA"
    sample_size: Optional[int] = None
    min_sale_value: float = 10_000
    max_sale_value: float = 50_000_000
    recent_sale_years: int = 5
    arms_length_only: bool = True


@dataclass
class ModelConfig:
    target: str = "VAL_TRANSFER"
    log_transform_target: bool = True
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    models_to_run: list = field(default_factory=lambda: [
        "lasso", "elastic_net", "bayesian_ridge", "lightgbm"
    ])
    lasso: dict = field(default_factory=dict)
    elastic_net: dict = field(default_factory=dict)
    bayesian_ridge: dict = field(default_factory=dict)
    lightgbm: dict = field(default_factory=dict)


@dataclass
class OutputConfig:
    directory: str = "results"
    save_predictions: bool = True
    save_model: bool = True
    generate_plots: bool = True


@dataclass
class AppConfig:
    snowflake: SnowflakeConfig = field(default_factory=SnowflakeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file.

    Looks for config.yaml in the project root by default.
    Environment variables prefixed with CPE_ override YAML values for credentials:
        CPE_SNOWFLAKE_ACCOUNT, CPE_SNOWFLAKE_USER, CPE_SNOWFLAKE_PASSWORD
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Copy config.yaml.example to config.yaml and fill in your credentials."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Build SnowflakeConfig with env-var overrides
    sf_raw = raw.get("snowflake", {})
    sf = SnowflakeConfig(
        account=os.environ.get("CPE_SNOWFLAKE_ACCOUNT", sf_raw.get("account", "")),
        user=os.environ.get("CPE_SNOWFLAKE_USER", sf_raw.get("user", "")),
        password=os.environ.get("CPE_SNOWFLAKE_PASSWORD", sf_raw.get("password")),
        authenticator=sf_raw.get("authenticator"),
        warehouse=sf_raw.get("warehouse", ""),
        database=sf_raw.get("database", "BUILDING_CARBON"),
        schema=sf_raw.get("schema", "PARCEL_DATA_2025Q1"),
        role=sf_raw.get("role"),
    )

    data_raw = raw.get("data", {})
    data = DataConfig(**{k: v for k, v in data_raw.items()})

    model_raw = raw.get("model", {})
    model = ModelConfig(**{k: v for k, v in model_raw.items()})

    output_raw = raw.get("output", {})
    output = OutputConfig(**{k: v for k, v in output_raw.items()})

    return AppConfig(snowflake=sf, data=data, model=model, output=output)


# ═══════════════════════════════════════════════════════════════════
# GLM-specific configuration (v4 pipeline)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GLMConfig:
    """
    Configuration for the statsmodels GLM with census tract fixed effects.

    Model: log(Sale_Price) ~ α_tract + β₁·SQFT + β₂·LOT_SIZE
           + γ_tract·SQFT + δ_tract·LOT_SIZE
           + β₃·BEDROOMS + β₄·BATHS + β₅·AGE
           + β₆·AC×CLIMATE + β₇·YEARS_SINCE_SALE
           + construction_dummies + ε

    With ~7,285 census tracts the design matrix has ~22K columns
    but is extremely sparse (~10 non-zeros per row of ~22K cols).
    """
    # --- Target ---
    target_col: str = "SALE_PRICE"
    log_target: bool = True

    # --- Continuous features (global coefficients) ---
    continuous_features: List[str] = field(default_factory=lambda: [
        "BUILDING_SQFT",
        "LOT_SIZE_AREA",
        "BEDROOMS",
        "TOTAL_BATHS_CALCULATED",
        "PROPERTY_AGE",
        "YEARS_SINCE_SALE",
    ])

    # --- Census tract fixed effects ---
    tract_col: str = "CENSUS_TRACT"
    # Tract-level interaction slopes (γ_tract·X, δ_tract·X)
    tract_interaction_features: List[str] = field(default_factory=lambda: [
        "BUILDING_SQFT",
        "LOT_SIZE_AREA",
    ])

    # --- Construction type dummies ---
    construction_col: str = "CONSTRUCTION_CODE_DESC"

    # --- AC × Climate interaction ---
    ac_col: str = "AIR_CONDITIONING_TYPE_DESC"

    # --- Filtering ---
    min_sale_price: float = 50_000.0
    max_sale_price: float = 20_000_000.0
    min_sqft: float = 100.0
    max_sqft: float = 25_000.0
    min_tract_sales: int = 10  # Tracts with fewer sales get pooled

    # --- Memory management ---
    prediction_chunk_size: int = 200_000
    sparse_dtype: str = "float32"

    # --- Output ---
    output_dir: Path = Path("results")
    model_artifact_dir: Path = Path("results/glm_artifacts")


# ═══════════════════════════════════════════════════════════════════
# Property type classification and model routing
# ═══════════════════════════════════════════════════════════════════

# USE_CODE_STD_LPS ranges → model group
PROPERTY_TYPE_RANGES: Dict[str, Tuple[int, int]] = {
    "residential":    (1000, 1099),
    "condo":          (1100, 1199),
    "multifamily":    (1200, 1399),
    "commercial":     (2000, 2999),
    "industrial":     (3000, 3999),
    "agricultural":   (4000, 4999),
    "vacant_land":    (5000, 5999),
    "institutional":  (6000, 6999),
    "recreation":     (7000, 7999),
}

# Which model handles each property type
MODEL_ROUTING: Dict[str, str] = {
    "residential":   "residential_v4",   # Stage 1: full tract FE GLM
    "condo":         "residential_v4",   # Merged into residential model
    "multifamily":   "nonres_glm",       # Stage 2: hierarchical GLM
    "commercial":    "nonres_glm",
    "industrial":    "nonres_glm",
    "recreation":    "nonres_glm",
    "vacant_land":   "land_model",       # LOT_SIZE × location model
    "agricultural":  "land_fallback",    # Land opportunity cost
    "institutional": "land_fallback",
}


def classify_property_type(use_code) -> str:
    """
    Classify a USE_CODE_STD_LPS value into a property type group.

    Returns one of: residential, condo, multifamily, commercial,
    industrial, agricultural, vacant_land, institutional, recreation,
    or 'unknown'.
    """
    try:
        code = int(float(use_code))
    except (ValueError, TypeError):
        return "unknown"
    for group_name, (lo, hi) in PROPERTY_TYPE_RANGES.items():
        if lo <= code <= hi:
            return group_name
    return "unknown"


def get_model_for_property_type(property_type: str) -> str:
    """Return the model name that handles this property type."""
    return MODEL_ROUTING.get(property_type, "land_fallback")


@dataclass
class NonResGLMConfig:
    """
    Configuration for Stage 2 non-residential GLM.

    Uses residential tract FE as a continuous location feature
    instead of fitting its own tract-level fixed effects (too
    sparse for non-res: commercial averages 3.6 sales/tract).

    County-level FE provide the location residual.
    """
    target_col: str = "SALE_PRICE"
    log_target: bool = True

    # Continuous features shared across non-res types
    continuous_features: List[str] = field(default_factory=lambda: [
        "BUILDING_SQFT",
        "LOT_SIZE_AREA",
        "PROPERTY_AGE",
        "YEARS_SINCE_SALE",
        "RES_TRACT_FE",       # From Stage 1 residential model
        "LOT_TO_BLDG_RATIO",
        "STORIES_NUM",
    ])

    # Property type dummies (reference = commercial)
    property_type_col: str = "PROPERTY_TYPE_GROUP"

    # County-level fixed effects (replaces tract FE)
    county_col: str = "COUNTYNAME"
    min_county_sales: int = 20

    # Building class dummies (44-55% populated for com/ind)
    bldg_class_col: str = "BLDG_CLASS"

    # No BEDROOMS/BATHS — meaningless for non-res (97%+ null)
    # No AC — only 12-16% populated for non-res
    # No construction dummies — only 44% populated

    # Filtering
    min_sale_price: float = 50_000.0
    max_sale_price: float = 50_000_000.0  # Higher ceiling for commercial
    min_sqft: float = 0.0       # Some parcels are land-only
    max_sqft: float = 500_000.0  # Warehouses can be huge
    min_county_sales: int = 20

    # Use codes included in non-res GLM training
    # Excludes ag/institutional (too few sales → land fallback)
    training_types: List[str] = field(default_factory=lambda: [
        "multifamily", "commercial", "industrial", "recreation",
    ])

    # Output
    output_dir: Path = Path("results")
    model_artifact_dir: Path = Path("results/nonres_artifacts")
