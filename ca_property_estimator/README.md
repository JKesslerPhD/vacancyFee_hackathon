# California Property Value Estimator

> **Vacancy Fee integration note:** this component was built for a separate
> program (below) and its model *training* (`src/pipeline.py`, `src/score_all.py`)
> runs against that program's Snowflake warehouse, which this repo has no
> access to and doesn't need. `scripts/export_vacancy_fee_estimates.py` takes
> the already-scored statewide output (`results/ca_all_parcels_estimated_values.parquet`,
> not committed — see that script's docstring) and reshapes the Sacramento
> subset into what `results/build_map_data.py` expects, replacing the retired
> `parcel_actualValue/` pipeline as this repo's market-value source.

**CARB AB 2446 Embodied Carbon Program**

Estimates current market values for all ~10.2 million California parcels,
disaggregated into land and improvement components. Improvement values feed
into the USEEIO model to estimate embodied carbon stocks in buildings.

## Why Not Just Use Assessed Values?
Under California's Proposition 13 (1978), a property's assessed value is locked
at the purchase price and may increase by at most 2% per year. Properties held
for decades can be assessed at 20–40% of market value. This makes raw assessed
values (`VAL_ASSD`) unreliable for estimating the current replacement cost of
California's building stock.

**Our solution:** Use recent arms-length sales (`VAL_TRANSFER`) as training data
where assessed values have been reset to market, then predict current market
values for all parcels using location and physical characteristics.

## Two Model Systems

### 1. Residential v4 GLM (`src/pipeline.py`)
A rich hedonic model for **residential + condo** parcels (~9M):
