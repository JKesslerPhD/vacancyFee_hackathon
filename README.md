# Sacramento Vacancy Fee — Data & Analysis

Supporting data, analysis, and reporting for [vacancyfee.org](https://vacancyfee.org/), a campaign to strengthen Sacramento's Vacant Lot and Vacant Building Monitoring and Enforcement Program. This repo turns county parcel and 311 service-call data into evidence for that policy fight: how many parcels sit vacant, what they're worth, and how vacancy shows up in the city's own complaint data.

## Objectives

1. **Identify vacant/underused parcels** in Sacramento County from assessor land-use codes, improvement values, and known vacant-use types (parking lots, abandoned service stations).
2. **Correlate vacancy with 311 service calls** (code enforcement, blight, homeless encampments) to show that unmanaged vacancy generates cascading costs to the city, not just to the parcel owner.
3. **Quantify the revenue case** for a graduated vacancy fee (Sacramento currently charges a flat $70/year regardless of parcel value) versus tiered fee structures used by comparable cities.
4. **Produce public-facing artifacts** — maps, charts, and a written report — usable in public comment, committee testimony, and social media, and low enough on the skill floor that hackathon volunteers (many first-time GIS/data users) could build on it in a single afternoon.

## Workflow

```
Raw county data (data/, not in git — see below)
        │
        ▼
scripts/build_hackathon_data.py  (external — produces hackathon_data/*)
        │
        ▼
hackathon_data/  ── trimmed CSV/GeoJSON/KML/GPKG + vacancy_tier classification
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                          ▼
hackathon_data/311_analysis.ipynb                       parcel_actualValue/run_pipeline.py
◄── PRIMARY ANALYSIS NOTEBOOK                           ◄── market-value estimation pipeline
  Step 1-2   load 311 calls (SacCounty_SalesForce311_       Estimates market value per parcel
             calls.gpkg), build a reproducible 50K-row     (CPI-deflated Prop 13 comps, hybrid
             beta subset for fast iteration                method), compares against assessed
  Step 3-7   clean data, join council districts +          value and an outside QC valuation,
             parcel attributes, engineer "likely            exports figures + a market-value
             vacancy indicator" flags (board-up,            shapefile.
             abandoned)
  Step 8-14  interactive maps (Folium), ZIP/street-
             level rankings, seasonality, category
             correlation analysis
  Step 15    binomial GLM: which 311 categories
             predict high-likelihood vacancy
             indicators
  Step 16-18 export outputs to hackathon_data/
             311_analysis_outputs/, assert artifacts
             were written, build the combined
             abandoned+blight heatmap
        │
        ▼
hackathon_data/311_analysis_outputs/   (checkpoints, CSVs, PNGs, HTML maps)
        │
        ├─────────────────────────────┬──────────────────────────────┐
        ▼                             ▼                              ▼
march17_vacant_lot_tax_       311_heatmap/                    maps/render_maps.py
support_report.qmd            vacancy_311_synthesis.py &      ◄── pyQGIS presentation map suite
◄── STANDALONE REPORT         predict_vacancy.py                County + midtown PNGs: 311
    ARTIFACT                  ◄── narrative-driven synthesis    density, vacant parcels by tier,
  Reformats the March 17      of vacancy × 311 (health &         311×vacancy synthesis, 311-
  committee public comment    safety burden per parcel) and      predicted candidate vacancies,
  into a Methods/Results/     a model that flags parcels          council districts.
  Discussion/Conclusion       that "look" vacant from 311
  report, reading directly    signals but aren't in the
  from the notebook's         coded-vacant set.
  checkpoint (step6_full_
  joined_checkpoint.gpkg)
  and exported figures.
  Renders to .html (Quarto).
                                                                              │
                                                                              ▼
                                                                 results/index.html
                                                                 ◄── "The Vacant Equity Gap" —
                                                                     static site bundling the
                                                                     headline figures + interactive
                                                                     map (map.js/map_data/) for
                                                                     public distribution.
```

The notebook is the source of truth for the 311/vacancy analysis; downstream artifacts (`.qmd` report, `311_heatmap/` synthesis, `maps/`, `results/`) read from its checkpoint/exports rather than re-deriving data independently, so re-running the notebook before rendering them keeps everything in sync.

## Sitemap

| Path | What it is |
|---|---|
| `hackathon_data/311_analysis.ipynb` | **Primary analysis notebook.** 311 calls × parcels × council districts → vacancy indicators, maps, GLM. |
| `hackathon_data/311_analysis_outputs/` | Notebook outputs: checkpoint GeoPackage, correlation CSVs/heatmaps, ZIP/street/time-series charts, Folium HTML maps, GLM interpretation. |
| `hackathon_data/reference_data/` | Council district shapefile and a parcel reference extract used as notebook join inputs. |
| `hackathon_data/starter_notebook.ipynb` | Simplified notebook for the hackathon's "Interactive Dashboard" track (loads data, basic maps/fee calcs). |
| `hackathon_data/style_*.qml` | QGIS layer styles for parcels by vacancy tier, land use, assessed value, lot size, owner. |
| `hackathon_data/SETUP_GUIDE.md` | QGIS / Google Earth Pro / Google My Maps / Python setup instructions for hackathon participants. |
| `hackathon_data/DATA_DOWNLOAD.md` | Google Drive links for the large hackathon data files (too big for git). |
| `311_heatmap/` | Vacancy × 311 synthesis (`vacancy_311_synthesis.py`, `predict_vacancy.py`, `correlation_analysis.py`) with the narrative writeup in `NARRATIVE.md`, signal reference in `ANALYSIS_NOTES.md`, and headline numbers in `findings.json`. Source of `public_comment_vacant_property_enforcement.txt`. |
| `maps/` | pyQGIS presentation map suite (`render_maps.py`, `prep_layers.py`) — county/midtown PNGs of 311 density, vacant parcels by tier, and the 311×vacancy synthesis; see `maps/README.md`. |
| `parcel_actualValue/` | Market-value estimation pipeline (`run_pipeline.py` runs estimate → hybrid estimate → visualize → QC comparisons → shapefile export) with CPI/Prop 13 deflator data and an external QC valuation cross-check. |
| `results/` | **"The Vacant Equity Gap"** — the public-facing static site (`index.html`, `map.js`, `map_data/`) bundling headline figures and an interactive map for distribution off this repo. |
| `march17_vacant_lot_tax_support_report.qmd` / `.html` | **Standalone report artifact.** Methods/Results/Discussion/Conclusion writeup built from the notebook's checkpoint, for the March 17 Law & Legislation Committee hearing. |
| `Resources/` | Source material for the March 17 hearing: the county's own program discussion memo (PDF), the submitted public comment, and committee responses. |
| `hackathon_plan.md` / `.pdf` | Hackathon run-of-show: schedule, five project tracks (story map, social kit, fee calculator, heat map, dashboard), data file guide. |
| `hackathon_questions.md` | Planning Q&A that shaped the hackathon's scope and the vacancy-tier definitions. |
| `storymap_idea.md` | Notes toward a public-facing story map layout (interactive heatmap, commercial/residential sections, walking tours, empty lots) — largely superseded by `results/`. |
| `311_data.gpkg`, `311_data_beta_50k.gpkg` | Raw and subsampled 311 call GeoPackages (duplicated at repo root and in `hackathon_data/`; large, git-ignored). |
| `resources.md` | Link to vacancyfee.org. |

Root-level data referenced in `CLAUDE.md` (`data/sac_county_parcel_assessors.gpkg`, `data/sacramento_identified_parcels.csv`, etc.) is the raw upstream source for `hackathon_data/` and is not committed to this repo — see `CLAUDE.md` for schema/join details and `hackathon_data/DATA_DOWNLOAD.md` for the Google Drive copy.

## Vacancy Classification

Parcels are tagged into three tiers (`vacancy_tier` column) — see `CLAUDE.md` for full rules:

- **Tier 1 — Coded Vacant** (19,364 parcels): land use code starts with "I".
- **Tier 2 — Zero Improvement** (9,151): $0 improvement value, excluding infrastructure/parks/agriculture/government.
- **Tier 3 — Parking/Abandoned** (155): parking lots and abandoned service stations.

`maps/` and `311_heatmap/` deliberately exclude Tier 2 from map output (only Tier 1 + Tier 3 shown) and avoid the term "blight" in favor of "health & safety / nuisance" — see `311_heatmap/NARRATIVE.md` for the reasoning.

## Status

- **Done:** vacancy-tier parcel classification; hackathon data exports (CSV/GeoJSON/KML/GPKG); the March 17, 2026 committee hearing and submitted written comment (`Resources/march17_vacant_lot_tax_support_comment.md`, committee responses in `Resources/march17_committee_responses.md`); the `311_analysis.ipynb` pipeline through GLM + exported outputs; the standalone `.qmd` report (rendered `.html` checked in); the `311_heatmap/` vacancy×311 synthesis and prediction model; the `maps/` pyQGIS presentation suite; the `parcel_actualValue/` market-value pipeline with QC cross-checks; and the `results/` public-facing site.
- **Not yet started / open:** no single automated pipeline ties `hackathon_data/311_analysis.ipynb` → `311_heatmap/` → `maps/` → `results/` together — each stage currently expects the previous stage's outputs to already exist and be re-run manually when upstream data changes.
