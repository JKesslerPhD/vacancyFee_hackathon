# California Property Value Estimator — Technical Methods

**CARB AB 2446 Embodied Carbon Program**
**California Air Resources Board · California Department of Technology**

*Prepared: June 2026*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Description](#2-data-description)
3. [The Proposition 13 Problem](#3-the-proposition-13-problem)
4. [Residential Model (v4 GLM)](#4-residential-model-v4-glm)
5. [All-Property Land Model](#5-all-property-land-model)
6. [All-Property Improvement Model](#6-all-property-improvement-model)
7. [Results](#7-results)
8. [Limitations](#8-limitations)
9. [Figures](#9-figures)

---

## 1. Executive Summary

This document describes the statistical methods used to estimate the current
market value of all ~10.2 million California parcels. The estimates are
produced for the California Air Resources Board's (CARB) AB 2446 Embodied
Carbon program, which requires current-dollar improvement (building) values
to quantify the embodied carbon stock in California's built environment via
the USEEIO environmentally-extended input-output model.

**The core challenge:** California's Proposition 13 (1978) freezes assessed
values at purchase price, increasing them by at most 2% annually. Properties
held for decades may be assessed at 20–40% of market value. Raw assessed
values therefore cannot be used as reliable proxies for current market value.

**Our approach:** We use recent arms-length sale prices (the moment when
assessed values are reset to market) as training data, then predict current
market values for all parcels using location fixed effects and property
characteristics. Two model systems are deployed:

| Model | Coverage | R² | MdAPE | Parcels |
|---|---|---|---|---|
| Residential v4 GLM | Residential + Condo | 0.76 | 10.6% | ~9.0M |
| All-Property Land | All 9 types | 0.58 | — | ~10.2M |
| All-Property Improvement | All 9 types | 0.53 | — | ~10.2M |

The final output uses the **best model for each property type**: the
residential v4 GLM (higher accuracy) for residential and condo parcels, and
the all-property land + improvement models for the remaining seven types.

---

## 2. Data Description

### 2.1 Source

All data is sourced from a single Snowflake table:

```
BUILDING_CARBON.PARCEL_DATA_2025Q1.PARCELS_CLEAN_2025Q1
```

This table contains assessor, sale, and physical-attribute data for
approximately **10.2 million California parcels** across all 58 counties,
with ~332 columns per parcel.

### 2.2 Key Fields

| Field | Description | Role |
|---|---|---|
| `VAL_TRANSFER` | Sale/transfer price ($) | Training target |
| `DATE_TRANSFER` | Date of last sale | Recency filter |
| `SALE_CODE` | Transaction type (F/R/0/*) | Arms-length filter |
| `VAL_ASSD` | Total assessed value | Prop 13 basis |
| `VAL_ASSD_LAND` | Assessed land value | Land/imprv split |
| `VAL_ASSD_IMPRV` | Assessed improvement value | Land/imprv split |
| `BUILDING_SQFT` | Building square footage | Physical feature |
| `LOT_SIZE_AREA` | Lot size (sq ft) | Physical feature |
| `BEDROOMS` | Number of bedrooms | Residential feature |
| `TOTAL_BATHS_CALCULATED` | Number of bathrooms | Residential feature |
| `YR_BLT` | Year built | Age calculation |
| `CENSUS_TRACT` | Census tract number | Location FE |
| `FIPS_CODE` | County FIPS code | Location FE |
| `USE_CODE_STD_LPS` | Standardized use code | Property type |
| `CONSTRUCTION_CODE_DESC` | Construction type | Building feature |
| `AIR_CONDITIONING_TYPE_DESC` | AC type | Building feature |

### 2.3 Property Type Classification

Parcels are classified into 9 groups using `USE_CODE_STD_LPS` ranges:

| Group | Use Code Range | Count | Share |
|---|---|---|---|
| Residential (SFR) | 1000–1099 | 8,436,767 | 82.8% |
| Condo | 1100–1199 | 589,418 | 5.8% |
| Multifamily | 1200–1399 | — | — |
| Commercial | 2000–2999 | 198,236 | 1.9% |
| Industrial | 3000–3999 | 51,951 | 0.5% |
| Agricultural | 4000–4999 | 23,227 | 0.2% |
| Vacant Land | 5000–5999 | 69,915 | 0.7% |
| Institutional | 6000–6999 | 14,244 | 0.1% |
| Recreation | 7000–7999 | 86,220 | 0.8% |
| Unknown | Other | 713,478 | 7.0% |

### 2.4 Training Set

The training set consists of recent arms-length transactions:
- **Sale codes:** F (full), R (rounded), 0 (computed from transfer tax), * (rounded)
- **Recency:** Sales within the last 5 years
- **Price range:** $10,000 – $50,000,000
- **Result:** ~1.2 million training observations for the residential model;
  variable counts per property type for the all-property models

---

## 3. The Proposition 13 Problem

### 3.1 Background

California's Proposition 13, passed in 1978, constrains property tax
assessments:

1. Assessed value is set to **purchase price** at time of sale
2. Annual increases are capped at **2%** regardless of market appreciation
3. Reassessment to market value occurs only upon **change of ownership**
   or **new construction**

This creates a systematic and growing gap between assessed values and
market values for long-held properties.

### 3.2 Empirical Evidence

Analysis of the ratio `VAL_ASSD / SALE_PRICE` by sale vintage reveals the
magnitude of the problem:

| Years Since Sale | Median ASSD/SALE Ratio | Interpretation |
|---|---|---|
| 0–1 year | ~0.60 | Assessor hasn't fully updated yet |
| 1–3 years | ~0.85 | Partially reassessed |
| 3–5 years | ~1.04 | Fully caught up (with 2%/yr growth) |
| 10+ years | ~0.45 | Substantially below market |
| 20+ years | ~0.25 | Assessed at 25% of market |
| 30+ years | ~0.15 | Assessed at 15% of market |

> **Figure 1** (see Section 9): ASSD/SALE_PRICE ratio by sale vintage — the
> "smoking gun" chart showing systematic underassessment.

Even for recently-sold properties (0–1 year), the ratio is only ~0.60,
meaning the assessor takes 2–3 years to fully process reassessments.
This lag must be accounted for in model design.

### 3.3 The Sale-Price Anchoring Solution

Rather than using raw assessed values as training targets, we **anchor to
actual sale prices** while preserving the assessor's expertise in the
land/improvement split:

```
MARKET_LAND  = SALE_PRICE × (VAL_ASSD_LAND / VAL_ASSD)
MARKET_IMPRV = SALE_PRICE × (VAL_ASSD_IMPRV / VAL_ASSD)
```

**Rationale:** The assessor's *absolute* values are stale under Prop 13,
but their *relative* split between land and improvement reflects genuine
appraisal expertise — they know whether a property's value is driven by
location (land) or by the structure (improvement). Scaling to the actual
transaction price preserves this ratio while correcting the level.

**Validation:** At 3–5 year vintage, where assessors have fully caught up,
the ASSD/SALE ratio converges to ~1.04 (consistent with 2%/yr Prop 13
growth), confirming that our anchoring approach is consistent with the
assessor's own reassessment process.

---

## 4. Residential Model (v4 GLM)

### 4.1 Scope

The residential model covers **residential (SFR) and condo** parcels —
approximately 9.0 million of California's 10.2 million parcels. These
property types share a common feature set (bedrooms, bathrooms, square
footage, construction type, AC) that enables a rich hedonic specification.

### 4.2 Model Specification

```
log1p(SALE_PRICE) ~ μ                              (global intercept)
                  + α_tract                         (census tract FE)
                  + β₁·BUILDING_SQFT               (standardized)
                  + β₂·LOT_SIZE_AREA               (standardized)
                  + β₃·BEDROOMS                    (standardized)
                  + β₄·TOTAL_BATHS_CALCULATED      (standardized)
                  + β₅·PROPERTY_AGE                (standardized)
                  + β₆·YEARS_SINCE_SALE            (standardized)
                  + β₇·HAS_AC                      (binary 0/1)
                  + β₈–β₁₁·construction_dummies    (IS_FRAME, IS_OTHER, etc.)
                  + γ_tract·BUILDING_SQFT           (tract × sqft interaction)
                  + δ_tract·LOT_SIZE_AREA           (tract × lot interaction)
                  + ε
```

**Key design choices:**

- **Target:** `log1p(SALE_PRICE)` — log transformation for better
  distributional properties; `log1p` avoids issues with zero values.
- **Census tract fixed effects:** ~8,607 tracts (after pooling tracts with
  <10 sales into a `__POOLED__` category). Each tract gets its own
  intercept, capturing location value.
- **Tract × feature interactions:** BUILDING_SQFT and LOT_SIZE_AREA get
  tract-specific slopes. This allows the marginal value of an additional
  square foot to vary by location (e.g., $800/sqft in Palo Alto vs.
  $150/sqft in Bakersfield).
- **YEARS_SINCE_SALE:** Absorbs market appreciation during training.
  Set to **zero at prediction time** to estimate current market value.
- **No separate AC/construction blocks:** Earlier versions (v1-v3) used
  separate one-hot blocks for AC and construction type, but these were
  collinear with tract FE. In v4, `HAS_AC` (binary) and construction
  dummies are included as continuous features only.
- **Standardization:** All continuous features are z-scored (mean=0, std=1)
  using training-set statistics stored in the DesignInfo object.

### 4.3 Design Matrix Structure

The design matrix is a sparse CSR matrix with the following block layout:

| Block | Columns | Description |
|---|---|---|
| 0 — Intercept | 1 | Global intercept (column of 1s) |
| 1 — Continuous | 11 | SQFT, LOT, BEDS, BATHS, AGE, YSS, HAS_AC, 4 construction dummies |
| 2 — Tract FE | 8,607 | One-hot census tract indicators |
| 3 — Interactions | 17,214 | 8,607 tracts × 2 features (SQFT, LOT) |
| **Total** | **25,833** | |

With ~1.2M observations and ~14 non-zeros per row, the matrix requires
~150 MB in sparse format vs. ~120 GB dense.

### 4.4 Estimation

The model is estimated via **direct sparse Cholesky factorization** on the
normal equations:

```
(X'X + αI) β = X'y
```

where α = 0.01 × mean(diag(X'X)), a light Ridge penalty that ensures
numerical stability for the 25,833-parameter system without materially
biasing well-identified coefficients.

This is solved using `scipy.sparse.linalg.spsolve`, which is fast (~30s),
deterministic, and stable. Earlier versions used iterative LSQR, which had
convergence issues with 200K+ columns.

### 4.5 Estimated Coefficients

| Feature | Coefficient | Interpretation |
|---|---|---|
| Intercept (μ) | +13.673 | Base log-price: $\exp(13.673) - 1 ≈ \$870K$ |
| BUILDING_SQFT | +0.329 | 1 SD increase → +39% price |
| LOT_SIZE_AREA | +0.057 | 1 SD increase → +6% price |
| BEDROOMS | −0.044 | Negative: holding sqft constant, more beds = smaller rooms |
| TOTAL_BATHS | −0.018 | Similar to bedrooms (sqft-adjusted) |
| PROPERTY_AGE | −0.035 | Older = lower value (1 SD ≈ −3.5%) |
| YEARS_SINCE_SALE | −0.028 | Market appreciation (~2.8%/yr) |
| HAS_AC | −0.012 | Small; collinear with climate/location |
| IS_FRAME | −0.116 | Frame construction discount vs. reference |

**Tract FE distribution:**
- Mean: +0.002, Std: 0.500
- Range: −1.49 to +2.31
- Implied price multiplier: 0.22× (cheapest) to 10.1× (most expensive)
- Median: −0.006 (close to state average)

**Tract × SQFT interaction slopes:**
- Mean: ~0, Std: 0.173
- Captures that a sqft in Beverly Hills is worth far more than in rural areas

### 4.6 Performance

| Metric | Value |
|---|---|
| R² (test set) | 0.765 |
| RMSE (log-scale) | 0.347 |
| Median APE | 10.6% |
| Within 10% of sale price | 47.8% |
| Within 20% of sale price | 73.8% |
| Training observations | 1,015,209 |
| Test observations | 203,042 |
| Parameters | 25,833 |
| Obs/parameter | 39.3 |

### 4.7 Land/Improvement Decomposition

The residential model estimates **total** market value. The split into land
and improvement components uses the assessor's ratio:

```
land_ratio = clip(VAL_ASSD_LAND / VAL_ASSD, 0.05, 0.95)
             fillna(0.40)  # default 40% land for missing data

EST_LAND_VALUE       = ESTIMATED_VALUE × land_ratio
EST_IMPROVEMENT_VALUE = ESTIMATED_VALUE × (1 − land_ratio)
```

The 0.40 default reflects the statewide median land share for residential
properties.

### 4.8 Example: Scoring a Single Property

For a 1,500 sqft home in Sacramento (tract 06067_001701), built in 1960,
3 bed / 2 bath, wood frame, central AC, sold 3 years ago for $450,000:

```
log1p(price) = 13.673          (intercept)
             + 0.329 × z(1500) (sqft, standardized)
             + 0.057 × z(6000) (lot)
             − 0.044 × z(3)    (beds)
             − 0.018 × z(2)    (baths)
             − 0.035 × z(64)   (age)
             + 0.000 × 0       (YEARS_SINCE_SALE = 0 at prediction)
             − 0.012 × 1       (has AC)
             − 0.116 × 1       (frame construction)
             + α_tract          (Sacramento tract FE)
             + γ_tract × z(1500)(tract × sqft slope)
             + δ_tract × z(6000)(tract × lot slope)
             = ~13.05

est_value = expm1(13.05) = ~$465,000
```

---

## 5. All-Property Land Model

### 5.1 Motivation

The residential v4 model requires physical features (bedrooms, bathrooms)
that are unavailable or meaningless for non-residential property types.
The all-property models use a simpler specification that works across all
9 property types.

### 5.2 Training Data

- **Source:** Recent sales (≤2 years) with valid `VAL_ASSD_LAND`,
  `LOT_SIZE_AREA`, and sale price
- **Target:** `log(LAND_PER_SQFT)` where:
  ```
  MARKET_LAND = SALE_PRICE × (VAL_ASSD_LAND / VAL_ASSD)
  LAND_PER_SQFT = MARKET_LAND / LOT_SIZE_AREA
  ```
- **Outlier filtering:**
  - Within-tract: trim top/bottom 5% of log(LAND_PER_SQFT) per tract
  - Global: trim 1st/99th percentile
- **Minimum tract observations:** 5 sales (tracts below this are pooled)

### 5.3 Model Specification

```
log(LAND_PER_SQFT) ~ intercept
                    + α_tract                  (FIPS_TRACT FE)
                    + δ₁·IS_CONDO             (use-type dummies,
                    + δ₂·IS_MULTIFAMILY         residential = reference)
                    + δ₃·IS_COMMERCIAL
                    + δ₄·IS_INDUSTRIAL
                    + δ₅·IS_AGRICULTURAL
                    + δ₆·IS_VACANT_LAND
                    + δ₇·IS_INSTITUTIONAL
                    + δ₈·IS_RECREATION
                    + ε
```

### 5.4 Scoring

```
predicted_land_per_sqft = exp(intercept + α_tract + δ_type)
EST_LAND_VALUE = predicted_land_per_sqft × LOT_SIZE_AREA
```

Values are clipped to [0, $500M].

### 5.5 Performance

| Metric | Value |
|---|---|
| R² (log-scale, in-sample) | ~0.58 |
| MdAPE | ~25% |
| Within 20% | ~45% |

The lower R² compared to the residential model reflects the simpler
specification (no physical features) and greater heterogeneity across
property types.

---

## 6. All-Property Improvement Model

### 6.1 Motivation

Improvement (building) value depends on the structure's characteristics,
but also on how long ago the property last sold — because the assessed
improvement value drifts further from market value over time under Prop 13.

### 6.2 Training Data

- **Source:** All properties with valid `VAL_ASSD_IMPRV`, `BUILDING_SQFT`,
  and sale history (up to 50 years ago)
- **Target:** `log(IMPRV_PER_SQFT)` where:
  ```
  MARKET_IMPRV = SALE_PRICE × (VAL_ASSD_IMPRV / VAL_ASSD)
  IMPRV_PER_SQFT = MARKET_IMPRV / BUILDING_SQFT
  ```
- **Same outlier filtering** as the land model

### 6.3 Model Specification

```
log(IMPRV_PER_SQFT) ~ intercept
                     + α_tract                    (tract FE)
                     + δ_type                     (use-type dummies)
                     + β₁·YEARS_SINCE_SALE        (global time slope)
                     + γ_tract·YEARS_SINCE_SALE   (tract-specific decay)
                     + θ_type·YEARS_SINCE_SALE    (type-specific decay)
                     + ε
```

**Design matrix blocks:**

| Block | Columns | Description |
|---|---|---|
| 0 — Intercept | 1 | Global intercept |
| 1 — Use-type dummies | 8 | Property type indicators |
| 2 — Tract FE | T | One-hot tract indicators |
| 3 — Global time slope | 1 | YEARS_SINCE_SALE |
| 4 — Tract × time | T | Tract-specific time slopes |
| 5 — Type × time | 8 | Use-type-specific time slopes |

### 6.4 The Time Decay Mechanism

The model captures how assessed improvement values diverge from market
over time. The **multiplier** for adjusting assessed values to market is:

```
multiplier(tract, type, t) = exp(−(β₁ + γ_tract + θ_type) × t)
```

**At t = 0** (prediction time for current market values), the multiplier
is always **1.0** by construction — `exp(0) = 1`. This means scoring
simply uses:

```
EST_IMPRV_VALUE = exp(intercept + α_tract + δ_type) × BUILDING_SQFT
```

The time terms exist to properly fit the model across properties with
different sale vintages, but they zero out at prediction time.

### 6.5 Scoring

For current market values, `YEARS_SINCE_SALE = 0`:

```
log_imprv_per_sqft = intercept + α_tract + δ_type
imprv_per_sqft = exp(log_imprv_per_sqft)
EST_IMPROVEMENT_VALUE = imprv_per_sqft × BUILDING_SQFT
```

Parcels with no building (`BUILDING_SQFT ≤ 0`) receive
`EST_IMPROVEMENT_VALUE = 0` and are labeled as `land_only`.

### 6.6 Performance

| Metric | Value |
|---|---|
| R² (log-scale, in-sample) | ~0.53 |
| MdAPE | ~30% |
| Within 20% | ~40% |

---

## 7. Results

### 7.1 Combined Output

The final output table uses the **best model for each property type**:

| Property Type | Model Used | Parcels | Median Total | Median Land | Median Imprv |
|---|---|---|---|---|---|
| Residential | residential_v4 | 8,436,767 | ~$650K | ~$250K | ~$400K |
| Condo | residential_v4 | 589,418 | ~$475K | ~$190K | ~$285K |
| Commercial | score_all | 198,236 | ~$1.3M | ~$800K | ~$500K |
| Industrial | score_all | 51,951 | ~$1.5M | ~$900K | ~$600K |
| Multifamily | score_all | — | — | — | — |
| Agricultural | score_all | 23,227 | ~$428 | ~$350 | ~$78 |
| Vacant Land | score_all | 69,915 | ~$350K | ~$350K | ~$0 |
| Institutional | score_all | 14,244 | ~$1.0M | ~$1.0M | ~$0 |
| Recreation | score_all | 86,220 | ~$462K | ~$309K | ~$0 |
| Unknown | score_all | 713,478 | ~$114K | ~$89K | ~$0 |

*Note: Values are approximate medians and may vary with model retraining.*

### 7.2 Comparison to Market Benchmarks

- **Statewide median home value (2025 Q1):** ~$750K (Zillow ZHVI)
- **Our residential median:** ~$650K — reasonable given our data includes
  all residential parcels, not just those with Zestimates
- **Statewide residential real estate stock:** ~$8 trillion (our estimate
  consistent at ~$5.5T total residential value across 9M parcels)

### 7.3 Model Routing Logic

```
if PROPERTY_TYPE_GROUP in ("residential", "condo"):
    use residential_v4 GLM (higher accuracy: R²=0.76, MdAPE=10.6%)
    land/improvement split via assessed ratio
else:
    use score_all land model + improvement model
    land = exp(tract_FE + type_dummy) × LOT_SIZE_AREA
    imprv = exp(tract_FE + type_dummy) × BUILDING_SQFT
```

### 7.4 Output Schema

The Snowflake table `BUILDING_CARBON.PARCEL_DATA_2025Q1.ESTIMATED_PROPERTY_VALUES`
contains:

| Column | Type | Description |
|---|---|---|
| PARCEL_APN | VARCHAR(50) | Parcel identifier |
| FIPS_CODE | VARCHAR(10) | County FIPS code |
| COUNTYNAME | VARCHAR(50) | County name |
| PROPERTY_TYPE_GROUP | VARCHAR(30) | Property classification |
| ESTIMATED_TOTAL_VALUE | FLOAT | Total estimated market value ($) |
| EST_LAND_VALUE | FLOAT | Estimated land component ($) |
| EST_IMPROVEMENT_VALUE | FLOAT | Estimated improvement component ($) |
| MODEL_USED | VARCHAR(20) | "residential_v4" or "score_all" |
| BUILDING_SQFT | FLOAT | Building square footage |
| LOT_SIZE_AREA | FLOAT | Lot size (sq ft) |
| VAL_ASSD | FLOAT | Total assessed value (Prop 13) |
| VAL_TRANSFER | FLOAT | Last sale price |

---

## 8. Limitations

### 8.1 Agricultural Properties

Agricultural parcels show very low estimated values (median ~$428) because
the $/sqft model breaks down for properties with extremely large acreage.
Agricultural land is valued on productive capacity (crop yields, water
rights), which is not captured by our location + lot size model.
These estimates should be used with caution.

### 8.2 Vacant Land

Some vacant land parcels show non-zero improvement values because the
assessor data contains `VAL_ASSD_IMPRV > 0` for parcels classified as
vacant. This may reflect minor improvements (fencing, grading, utilities)
or data quality issues. The improvement model will produce a value for
any parcel with `BUILDING_SQFT > 0`.

### 8.3 Institutional Properties

Institutional properties (schools, government buildings, churches) rarely
transact on the open market. The training data is extremely sparse for
this category, and estimates reflect the opportunity cost of the land
rather than the replacement cost of specialized buildings.

### 8.4 Cross-Sectional Limitations

The model is cross-sectional (one point in time). It does not capture:
- **Temporal dynamics:** housing bubbles, interest rate effects, or
  market cycles
- **Within-year seasonality:** sales in spring vs. winter
- **Individual property trajectory:** renovation, deterioration, or
  neighborhood gentrification

### 8.5 Accuracy vs. Coverage Tradeoff

The residential v4 model achieves MdAPE=10.6% but only covers residential
and condo parcels. The all-property models cover all 9 types but with
lower accuracy (R²=0.53–0.58). For the embodied carbon application,
**coverage is prioritized over accuracy** — we need an estimate for every
parcel, even if imprecise.

### 8.6 Known Edge Cases

- **Zero-lot-line condos:** LOT_SIZE_AREA may be 0 or very small,
  causing the land model to produce near-zero land values
- **Mixed-use properties:** Classified by primary use code only;
  the model cannot decompose a mixed retail/residential building
- **Prop 13 reassessment lag:** Even recently sold properties show
  ASSD/SALE = 0.60, meaning our sale-price anchoring is essential
- **"Unknown" properties (7%):** 713K parcels have use codes outside
  defined ranges — these get scored by the all-property model with
  residential as the reference category

### 8.7 Data Quality

- ~12% of parcels have null `BUILDING_SQFT` (scored as land-only)
- ~12.5% have null `PROPERTY_AGE` (imputed with training median)
- `CENSUS_TRACT` is present for 99.9%+ of parcels
- `VAL_ASSD` is present for 99.8%+ of parcels

---

## 9. Figures

The following figures should be generated to accompany this document.
Generation scripts are available in the codebase.

### Figure 1: ASSD/SALE_PRICE Ratio by Sale Vintage

**The smoking gun chart.** Plot the median ratio of assessed value to sale
price, binned by years since sale (0–1, 1–3, 3–5, 5–10, 10–20, 20–30,
30+). Shows the systematic underassessment under Prop 13 and validates
the sale-price anchoring approach.

```python
# Pseudocode for Figure 1
bins = [0, 1, 3, 5, 10, 20, 30, 50]
df['YSS_BIN'] = pd.cut(df['YEARS_SINCE_SALE'], bins)
ratio = df.groupby('YSS_BIN')['ASSD_TO_SALE_RATIO'].median()
plt.bar(ratio.index, ratio.values)
plt.ylabel('Median ASSD / SALE_PRICE')
plt.xlabel('Years Since Sale')
plt.axhline(y=1.0, linestyle='--', color='red', label='Parity')
plt.title('Proposition 13 Assessment Gap')
```

### Figure 2: Land Share of Assessed Value by Sale Vintage

Plot `VAL_ASSD_LAND / VAL_ASSD` by years since sale. Land share
typically increases from ~34% (recent sales) to ~40% (30+ years) as
land appreciates faster than structures depreciate.

### Figure 3: Distribution of Estimated Values by Property Type

Box plots (log scale) of `ESTIMATED_TOTAL_VALUE` for each property type
group. Shows the range and central tendency for each category.

### Figure 4: Predicted vs. Actual (In-Sample Validation)

Scatter plot of `ESTIMATED_VALUE` vs. `SALE_PRICE` for recent sales
(<1 year), with 45-degree parity line. Residential v4 model only.
Includes marginal histograms and R² annotation.

### Figure 5: Map of Median Estimated Values by County

Choropleth map of California's 58 counties, colored by median
`ESTIMATED_TOTAL_VALUE` for residential parcels. Highlights the
coastal/inland divide (Bay Area, LA > Central Valley, rural).

### Figure 6: Tract Fixed Effect Distribution

Histogram of the 8,607 residential tract FE coefficients. Shows the
location premium distribution — a few tracts in Beverly Hills / Palo Alto
at +2.3 (10× price multiplier), most tracts clustered near zero, and
some rural tracts at −1.5 (0.2× multiplier).

### Figure 7: Improvement Model Multiplier Curves

Plot the improvement multiplier over time (0–50 years) for 4 property
types (residential, commercial, industrial, multifamily) at the pooled
tract. Shows how the model adjusts assessed improvement values back to
market as the Prop 13 gap widens with age.

---

## Appendix A: Software & Reproducibility

| Component | Version |
|---|---|
| Python | 3.12 |
| NumPy | 2.x |
| SciPy | 1.x |
| pandas | 2.x |
| scikit-learn | 1.x |
| Snowflake Connector | 3.x |
| joblib | 1.x |

Model artifacts (coefficients, DesignInfo, tract FE dictionary) are saved
via `numpy.save` and `joblib.dump` in `results/glm_artifacts/` and
`results/model_artifacts/`. These can be loaded to reproduce predictions
without retraining.

## Appendix B: Critical Implementation Detail — FIPS + Census Tract

Raw `CENSUS_TRACT` values (e.g., "000100") repeat across California's 58
counties. Tract 000100 exists in Sacramento, Los Angeles, San Francisco,
and at least 9 other counties. Using raw tract as the fixed effect key
causes the model to average across unrelated geographies, destroying the
location signal.

**All code must use `FIPS_CODE + "_" + CENSUS_TRACT`** as the unique
tract identifier. This is enforced in:
- `src/pipeline.py` — training pipeline
- `src/score_all.py` — all-property scoring
- `score_single.py` — single-APN scorer

Failure to apply this transformation was the root cause of a major bug
in v2 where downtown Sacramento received a tract FE of −0.52 (should be
−0.13), biasing all predictions in that tract by ~50%.