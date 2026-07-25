# `results/` — public-facing pages

Two static pages, both meant to be hosted (not opened via `file://` — they
`fetch()` their data files, which browsers block from `file://` due to CORS)
and both embeddable via `<iframe>`:

- **`index.html`** — "The Vacant Equity Gap." Headline figures + interactive
  parcel map. Built by `build_map_data.py`.
- **`vacancy_explorer.html`** — Census block group vacancy choropleth +
  council district outlines + revenue-impact overlay, City of Sacramento
  only. Built by `build_vacancy_explorer_data.py`.

## Rebuilding `vacancy_explorer.html`'s data

```bash
# 1. hackathon_data/vacant_parcels_qc.csv must exist (hackathon_data/qc_park_exclusion.py)
# 2. ca_property_estimator/results/parcels_market_value_estimated.csv must exist
#    (ca_property_estimator/scripts/export_vacancy_fee_estimates.py)
# 3. revenue_impact/results/{parcel_revenue_estimates,district_revenue_summary}.csv
python revenue_impact/estimate_lost_revenue.py

# 4. downloads Census block group boundaries, clips to city limits, aggregates
python results/build_vacancy_explorer_data.py
```

Writes `map_data/block_groups_vacancy.json` and
`map_data/council_districts_revenue.json` — both tracked (small, a few
hundred KB), unlike the raw per-parcel CSVs elsewhere in this repo.

## Embedding on vacancyfee.org

Both pages need to be hosted somewhere (GitHub Pages off this repo, or
copied onto the main site) — `vacancy_explorer.html` fetches its two JSON
files by relative path, so keep `map_data/` alongside it wherever it's
deployed. Then:

```html
<iframe
  src="https://<your-host>/vacancy_explorer.html"
  style="width: 100%; height: 640px; border: 0;"
  loading="lazy"
  title="Sacramento vacant parcel explorer">
</iframe>
```

Give the iframe an explicit height (the page uses `100vh`, which resolves
against the iframe's own box, not the parent page — a collapsed/auto-height
iframe will show a 0px-tall map). 640px comfortably fits the legend and
control panel; go taller on desktop-first layouts if you have the space.

**Not visually verified in-browser** — I don't have a way to render or
screenshot pages in this environment, only serve and curl them. I checked
the page over HTTP (200s on the HTML and both JSON files, valid GeoJSON
structure, property names cross-checked against what the Python build script
writes) but you should open it yourself before publishing.
