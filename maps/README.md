# Map suite — Vacancy × 311 (pyQGIS)

A small suite of presentation maps over an OpenStreetMap basemap, built with
pyQGIS for a local activism presentation.

> **Terminology:** what older drafts called "blight" is framed here as
> **health & safety / nuisance** 311 calls — urban-land-use research avoids
> "blight" for its racist urban-renewal connotations.

## Figures (`figures/`)

| File | What it shows |
|---|---|
| `health_safety_311_county.png`  | Health & safety 311-call density across the urbanized county |
| `health_safety_311_midtown.png` | Same density zoomed to downtown / midtown |
| `vacant_parcels_county.png`     | 28,324 vacant parcels coloured by vacancy tier, county |
| `vacant_parcels_midtown.png`    | Vacant parcels by tier across the central grid |
| `synthesis_county.png`          | H&S density + vacant parcels outlined, county |
| `synthesis_midtown.png`         | H&S density + vacant parcels outlined, midtown |
| `predicted_vacancy_county.png`  | 311-predicted candidate vacancies (beyond the coded set), county |
| `predicted_vacancy_midtown.png` | 311-predicted candidate vacancies, midtown |
| `council_districts.png`         | The eight Sacramento city council districts, numbered |

**Cartography.** The basemap is dropped 15% opacity so the data leads; the
Sacramento County boundary is drawn on every county map; every map carries a
scale bar and north arrow with minimal chrome. **Zero-improvement (Tier 2)
parcels are excluded from every map** — only coded-vacant (Tier 1) and
parking/abandoned (Tier 3) are shown.

The "311 heatmap" is built as a smoothed density **raster** (numpy + GDAL), not
QGIS's live heatmap renderer: the renderer's auto-scaling is dominated by a
~4,400-call coordinate pile-up (a default geocode point) that washes the real
clusters out to transparent. Building the grid ourselves lets us clip that
outlier and control the smoothing.

The predicted-vacancy maps come from `../311_heatmap/predict_vacancy.py`, which
scores every parcel by its 311 signal profile and flags high-scoring parcels
that are **not** already in the coded-vacant set.

## Rebuilding

```bash
# 1. export the health & safety 311 points (default python, needs geopandas)
python maps/prep_layers.py

# 2. (optional) build the 311-predicted candidate vacancies
python 311_heatmap/predict_vacancy.py

# 3. render the maps (QGIS python — has qgis.core + osgeo)
"C:\Program Files\QGIS 3.38.1\bin\python-qgis.bat" maps\render_maps.py
```

**On macOS**, `render_maps.py` needs QGIS's own bundled Python (has `qgis.core`
built in — a `pip install qgis` into your regular interpreter won't work).
Install the **LTR (3.x)** line, not the latest 4.x: this script uses PyQt5-style
flat enums (`Qt.AlignLeft`) that PyQt6/Qt6 in QGIS 4.x renamed to nested
enums (`Qt.AlignmentFlag.AlignLeft`), so it'll crash under 4.x.

```bash
brew install --cask qgis@ltr        # QGIS-LTR.app, ~3.4 GB

export PYTHONHOME="/Applications/QGIS-LTR.app/Contents/Frameworks"
export PROJ_DATA="/Applications/QGIS-LTR.app/Contents/Resources/qgis/proj"
export PROJ_LIB="$PROJ_DATA"
# Must be the .app bundle root, NOT Contents/MacOS -- QGIS's macOS plugin-path
# derivation appends "Contents/PlugIns/qgis" unconditionally, so a
# MacOS-suffixed prefix silently doubles "Contents" and leaves every
# non-core provider (including "wms", which the XYZ basemap needs)
# unregistered with no error until the layer fails to load.
export QGIS_PREFIX_PATH="/Applications/QGIS-LTR.app"

"/Applications/QGIS-LTR.app/Contents/MacOS/python3.12" maps/render_maps.py
```

Inputs: `data/SacCounty_SalesForce311_calls.gpkg` (on the project Google Drive),
`hackathon_data/vacant_parcels.geojson`, and the council-district shapefile.
`prep_layers.py` also fetches the county boundary and reprojects the districts to
WGS84 (`maps/data/sacramento_county.geojson`, `council_districts.geojson` — small
and tracked). Heavy intermediates (`hs_311.gpkg`, `density_*.tif`,
`predicted_vacancies.gpkg`) are regenerated and git-ignored; only the scripts,
the small boundary layers, and the PNG figures are tracked.
