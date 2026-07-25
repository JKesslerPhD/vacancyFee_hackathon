"""
Snowflake data loader for the CA Property Estimator.

Connects to Snowflake, pulls parcel data, and applies initial filters
to identify the training set (recently sold, arms-length transactions).
"""

import logging
from typing import Optional

import pandas as pd
import snowflake.connector

from .config import AppConfig

logger = logging.getLogger(__name__)


def get_snowflake_connection(cfg: AppConfig) -> snowflake.connector.SnowflakeConnection:
    """Create and return a Snowflake connection."""
    sf = cfg.snowflake
    connect_params = {
        "account": sf.account,
        "user": sf.user,
        "warehouse": sf.warehouse,
        "database": sf.database,
        "schema": sf.schema,
    }
    if sf.password:
        connect_params["password"] = sf.password
    if sf.authenticator:
        connect_params["authenticator"] = sf.authenticator
    if sf.role:
        connect_params["role"] = sf.role

    logger.info("Connecting to Snowflake account: %s", sf.account)
    return snowflake.connector.connect(**connect_params)


def build_query(cfg: AppConfig) -> str:
    """
    Build the SQL query to pull parcel data for modeling.

    Selects the columns needed for feature engineering and filters
    to California properties with valid data.
    """
    dc = cfg.data
    fqn = f"{cfg.snowflake.database}.{cfg.snowflake.schema}.{dc.table}"

    # ── Columns to retrieve ──────────────────────────────────────────
    columns = """
        PARCEL_APN,
        FIPS_CODE,
        COUNTYNAME,
        SITE_ADDR,
        SITE_CITY,
        SITE_ZIP,
        LATITUDE,
        LONGITUDE,
        CENSUS_TRACT,

        -- Assessment values (Prop 13 basis)
        ASMT_YEAR,
        VAL_ASSD_LAND,
        VAL_ASSD_IMPRV,
        VAL_ASSD,

        -- Market values from assessor
        VAL_MRKT_LAND,
        VAL_MRKT_IMPRV,
        VAL_MARKET,
        YR_MRKT_VAL,

        -- Sale / transfer data (target variable source)
        ASMT_RCDRS_DATE_TRANSFER,
        ASMT_VAL_TRANSFER,
        ASMT_SALE_CODE,
        LAST_SALE_DATE_TRANSFER,
        LAST_DATE_TRANSFER,
        LAST_SALE_SALE_CODE,
        LAST_SALE_FULL_PART_CODE,
        ARMS_LENGTH_FLAG,
        DATE_TRANSFER,
        VAL_TRANSFER,
        SALE_CODE,
        SALE_CODE_DESC,

        -- Prior sale for appreciation calc
        PRIOR_SALE_DATE_TRANSFER,
        PRIOR_SALE_VAL_TRANSFER,
        PRIOR_SALE_SALE_CODE,

        -- Physical property attributes
        USE_CODE_STD_LPS,
        USE_CODE_STD_DESC_LPS,
        USE_CODE_MUNI_DESC,
        BUILDING_TYPE_SIMPLIFIED,
        BUILDING_TYPE_DETAIL,
        BLDG_CLASS,
        STYLE_DESC,
        CONSTRUCTION_CODE_DESC,
        EXTERIOR_WALL_DESC,
        FOUNDATION_TYPE,
        ROOF_COVER_DESC,
        HEATING_DESC,
        AIR_CONDITIONING_TYPE_DESC,
        FIREPLACE_DESC,
        POOL_CODE_DESC,
        GARAGE_CODE_DESC,
        PARKING_TYPE,
        FLOOR_TYPE,
        BSMT_1_CODE,
        BASEMENT_FINISH_DESC,

        -- Quantitative features
        BUILDING_SQFT,
        LIVING_SQFT,
        LOT_SIZE_AREA,
        LOT_SIZE_AREA_UNIT,
        ASSR_SQFT,
        ASSR_ACREAGE,
        UNITS_NUMBER,
        TOTAL_ROOMS,
        BEDROOMS,
        TOTAL_BATHS_CALCULATED,
        NUMBER_OF_BATHS,
        PARTIAL_BATHS,
        STORIES_NUMBER,
        PARKING_SPACES,
        YR_BLT,
        YR_BLT_EFFECT,

        -- Lot dimensions
        LOT_WIDTH,
        LOT_DEPTH,
        SHAPE_AREA,

        -- Owner / tax info
        OWNER_OCCUPIED,
        CA_HOME_OWNERS_EXEMPT,
        TAX_AMOUNT,
        TAX_YEAR,
        TAX_RATE_CODE_AREA,

        -- Spatial indices
        H3_RES5,
        H3_RES6,
        H3_RES7,
        H3_RES8,
        H3_RES9,

        -- Loan data (for market signal)
        LOAN_TYPE_1,
        LOAN_TYPE_1_DESC,

        -- AVM baseline
        AVM_VALUE,

        -- Duplicate flag
        DUPLICATE_SUMS
    """

    query = f"""
    SELECT
        {columns}
    FROM {fqn}
    WHERE SITE_STATE = '{dc.state_filter}'
      AND DUPLICATE_SUMS = FALSE
    """

    if dc.sample_size:
        query += f"\n    ORDER BY RANDOM()\n    LIMIT {dc.sample_size}"

    return query


def load_data(cfg: AppConfig) -> pd.DataFrame:
    """
    Load parcel data from Snowflake into a pandas DataFrame.

    Returns the raw data with minimal type conversions.
    """
    conn = get_snowflake_connection(cfg)
    query = build_query(cfg)
    logger.info("Executing query (sample_size=%s)...", cfg.data.sample_size)

    try:
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    finally:
        conn.close()

    return df


def identify_training_set(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """
    Flag rows that qualify as training data.

    Training properties are those with:
      - A valid, recent arms-length sale
      - Transfer value within configured min/max bounds
      - Full (not partial) interest transfer

    Adds columns:
      - IS_TRAINING: bool
      - SALE_DATE_PARSED: datetime
      - SALE_PRICE: float (cleaned VAL_TRANSFER)
    """
    dc = cfg.data
    df = df.copy()

    # ── Parse sale date ──────────────────────────────────────────────
    # Try multiple date columns in priority order
    date_cols = ["DATE_TRANSFER", "LAST_SALE_DATE_TRANSFER", "ASMT_RCDRS_DATE_TRANSFER"]
    df["SALE_DATE_PARSED"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in date_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            mask = df["SALE_DATE_PARSED"].isna() & parsed.notna()
            df.loc[mask, "SALE_DATE_PARSED"] = parsed[mask]

    # ── Parse sale price ─────────────────────────────────────────────
    df["SALE_PRICE"] = pd.to_numeric(df["VAL_TRANSFER"], errors="coerce")

    # ── Determine cutoff date ────────────────────────────────────────
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=dc.recent_sale_years)

    # ── Build training mask ──────────────────────────────────────────
    mask = (
        df["SALE_DATE_PARSED"].notna()
        & (df["SALE_DATE_PARSED"] >= cutoff)
        & df["SALE_PRICE"].notna()
        & (df["SALE_PRICE"] >= dc.min_sale_value)
        & (df["SALE_PRICE"] <= dc.max_sale_value)
    )

    if dc.arms_length_only:
        # Use SALE_CODE to identify market-rate transactions
        # F = Full amount computed
        # R = Rounded by county (still a real sale)
        # 0 = Full amount computed from transfer tax
        # * = Sales price or transfer tax rounded
        arms_length_codes = ["F", "R", "0", "*"]
        if "SALE_CODE" in df.columns:
            sale_code = df["SALE_CODE"].astype(str).str.strip().str.upper()
            arms_mask = sale_code.isin(arms_length_codes)
            # Also accept if ARMS_LENGTH_FLAG is set (may be populated in some counties)
            if "ARMS_LENGTH_FLAG" in df.columns:
                alf = df["ARMS_LENGTH_FLAG"].astype(str).str.upper()
                arms_mask = arms_mask | alf.isin(["Y", "1", "TRUE"])
            mask = mask & arms_mask
            logger.info("Arms-length filter: %d rows pass (codes: %s)",
                        arms_mask.sum(), arms_length_codes)

    df["IS_TRAINING"] = mask
    logger.info(
        "Training set: %d of %d rows (%.1f%%)",
        mask.sum(), len(df), 100 * mask.sum() / max(len(df), 1),
    )

    return df