# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GIS/data analysis project for identifying vacant parcels in Sacramento County, California, in support of vacancy fee policy research (see https://vacancyfee.org/). The project is data-driven with no application code — it consists of geospatial and assessor datasets for parcel-level analysis.

## Data Files

All data lives in `data/`. Files are large (total ~1.3 GB) and should not be committed to version control carelessly.

### Core Datasets

- **`sac_county_parcel_assessors.gpkg`** (198 MB) — GeoPackage with a single layer `Parcels` containing 482,403 Sacramento County parcel geometries and assessor attributes. CRS: EPSG:2226 (California State Plane Zone 2, US feet). Key columns: `APN`, `LOT_SIZE`, `LANDUSE`, `LU_GENERAL`, `LU_SPECIF`, `LU_DETAIL`, `CITY`, `ZIP`, `geometry`.

- **`sacramento_identified_parcels.csv`** (986 MB, ~486K rows) — Comprehensive parcel-level dataset with 270+ columns including ownership, assessed values, sales history, building characteristics, mortgage info, and H3 spatial indices. Join key to GeoPackage: `PARCEL_APN` ↔ `APN` (14-digit format like `22902620010000`).

### Supplemental Data (zipped)

- **`2025_Secured_Public_Roll_Excel*.zip`** — Sacramento County 2025 secured property tax roll (87 MB xlsx + layout doc)
- **`2025_Unsecured_Public_Roll_Excel*.zip`** — 2025 unsecured tax roll (5.8 MB xlsx + layout doc)
- **`2year Transfer List*.zip`** — Two-year property transfer list (nested zip)

## Environment

- Python 3.13.1 with `geopandas 1.0.1` available
- GDAL/OGR CLI tools are **not** installed; use Python (geopandas, fiona, shapely) for geospatial operations
- GeoPackage can also be queried directly via `sqlite3` for non-spatial queries

## Common Operations

```bash
# Read GeoPackage
python -c "import geopandas as gpd; gdf = gpd.read_file('data/sac_county_parcel_assessors.gpkg')"

# Query GeoPackage via SQL (fast for non-spatial lookups)
python -c "import sqlite3; conn = sqlite3.connect('data/sac_county_parcel_assessors.gpkg'); print(conn.execute('SELECT COUNT(*) FROM Parcels').fetchone())"

# Read large CSV in chunks (986 MB — avoid loading entirely into memory at once)
python -c "import pandas as pd; chunks = pd.read_csv('data/sacramento_identified_parcels.csv', chunksize=50000)"
```

### Additional GIS Data

- **`SacCounty_SalesForce311_calls.gpkg`** (783 MB) — 311 call data for correlation with vacant areas
- **`quickOSM_buildings_2025_07_01.gpkg`** (130 MB) — OSM building footprints
- **`quickOSM_roads_2025_07_01.gpkg`** (43 MB) — OSM road network
- **`council_districts/`** — Sacramento council district boundary shapefile
- **`LandUseCodeQuickReference.pdf`** — Sacramento County land use code reference

## Hackathon Data (`hackathon_data/`)

Pre-processed datasets built by `scripts/build_hackathon_data.py`:

- **`vacant_parcels.csv`** (9 MB, 28,670 rows) — Vacant/underused parcels with 40 key columns and `vacancy_tier` classification
- **`parcels_trimmed.csv`** (166 MB) — All 486K parcels, 40 key columns
- **`vacant_parcels.geojson`** (17 MB) — Vacant parcels with geometry (WGS84)
- **`vacant_parcels.kml`** (14 MB) — For Google Earth Pro
- **`parcels_simplified.gpkg`** (262 MB) — All parcels with simplified geometry + joined attributes
- **`starter_notebook.ipynb`** — Jupyter notebook with data loading, maps, fee calculations
- **`SETUP_GUIDE.md`** — QGIS, Google Earth Pro, Google My Maps setup instructions

### Vacancy Classification

Parcels are classified into three tiers in `vacancy_tier`:
- **Tier 1: Coded Vacant** (19,364) — Land use code starts with "I" per Sacramento County scheme
- **Tier 2: Zero Improvement** (9,151) — $0 improvement value, excluding infrastructure/parks/agriculture/government
- **Tier 3: Parking/Abandoned** (155) — Parking lots (BFH) and abandoned service stations (BFK)

The Tier 2 park exclusion above isn't actually applied in `build_hackathon_data.py`: publicly-owned park land is tax-exempt so it has $0 improvement value and matches the heuristic anyway. Separately, `VAL_ASSD_IMPRV` is null (not literally $0) for every Tier 2 row, so any occupied structure with missing improvement data gets swept in too — 1,290 parcels show direct structural evidence (sqft/bedrooms/year-built), 646 of them coded `SINGLE FAMILY RESIDENTIAL`. A third pass catches tax-exempt public infrastructure the structural check still misses (every field null, not just improvement value) — 113 parcels coded `PARKING GARAGE, PARKING STRUCTURE` / `AIRPORT & RELATED` / `CEMETERY (EXEMPT)`, all government or district owned. `hackathon_data/qc_vacancy_exclusions.py` is a downstream QC pass implementing all three checks — 1,647 parcels total (244 park-owned, 1,290 occupied-structure, 113 active-infrastructure) — and writes corrected `vacant_parcels_qc.csv`/`.geojson`, which `311_heatmap/`, `maps/`, and `revenue_impact/` prefer automatically when present. It's a patch on the output, not a fix to the classification script.

## Key Considerations

- The raw CSV is ~1 GB. Use `chunksize` with pandas or filter columns with `usecols` to avoid memory issues. The trimmed version (`hackathon_data/parcels_trimmed.csv`) is 166 MB and loads faster.
- APN formats differ: the GeoPackage `APN` column is string type; the CSV `PARCEL_APN` may load as int64. Cast both to string before joining.
- The GeoPackage CRS is EPSG:2226 (US feet). Convert to EPSG:4326 for lat/lon or web mapping.
- Land use classification is hierarchical: `LANDUSE` (code) → `LU_GENERAL` → `LU_SPECIF` → `LU_DETAIL`. Codes starting with "I" indicate vacant land (see `LandUseCodeQuickReference.pdf`).
