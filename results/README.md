# `results/` — public-facing pages

Two static pages, both embeddable via `<iframe>`:

- **`index.html`** — "The Vacant Equity Gap." Headline figures + interactive
  parcel map. Built by `build_map_data.py`. Needs to be hosted (not opened
  via `file://`) — it `fetch()`es its data files from `map_data/`, which
  Chrome/Safari block from a `file://` page under CORS.
- **`vacancy_explorer.html`** — "Potential Benefits of a Vacancy Fee": a
  property-count choropleth (block groups zoomed out, real census blocks
  zoomed in past z15) + council district outlines + revenue overlay, City
  of Sacramento only. Built by `build_vacancy_explorer_data.py` from
  `vacancy_explorer_template.html` **with block groups + districts +
  citywide totals inlined** — so this opens by double-clicking it, no
  server needed, in addition to being hostable/embeddable as-is. The
  census-block layer is the exception: it's ~5.6MB, so it's fetched lazily
  from `map_data/blocks_vacancy.json` only once the map is zoomed in, and
  only works when the page is served (not opened via `file://`) — the core
  experience (block groups, districts, callout) works either way. Edit the
  *template*, not `vacancy_explorer.html` directly — it's regenerated (and
  overwritten) on every build.

## Rebuilding `vacancy_explorer.html`

```bash
# 1. hackathon_data/vacant_parcels_qc.csv must exist (hackathon_data/qc_vacancy_exclusions.py)
# 2. ca_property_estimator/results/parcels_market_value_estimated.csv must exist
#    (ca_property_estimator/scripts/export_vacancy_fee_estimates.py)
# 3. revenue_impact/results/{parcel_revenue_estimates,district_revenue_summary}.csv
python revenue_impact/estimate_lost_revenue.py

# 4. downloads Census block-group AND real census-block (TABBLOCK20, ~350MB
#    statewide, filtered locally) boundaries, clips to city limits,
#    aggregates onto both geographies + districts, writes map_data/*.json,
#    and regenerates vacancy_explorer.html with block groups/districts inlined
python results/build_vacancy_explorer_data.py
```

`map_data/block_groups_vacancy.json` and `map_data/council_districts_revenue.json`
are also written standalone and tracked (small, a few hundred KB each) — not
used by `vacancy_explorer.html` itself (which has its own inlined copy) but
handy for debugging or reuse elsewhere without re-running the full build.
`map_data/blocks_vacancy.json` (~5.6MB, 7,269 real census blocks) is tracked
too, but is fetched at runtime, not inlined -- see above.

## Embedding on vacancyfee.org

`vacancy_explorer.html` is fully self-contained (Leaflet + OSM tile URLs load
from CDN over the network; the vacancy/revenue data itself is inlined) —
host the one file wherever you like, `index.html` needs its whole directory
(including `map_data/`) deployed together. Either way:

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

**Verified with a headless-Chromium (Playwright) screenshot pass**, both
served over HTTP and opened directly via `file://`: zero console errors
either way, choropleth and district outlines render against the OSM
basemap, the metric dropdown reshades the map, the district-outline toggle
works, and a hovered district's tooltip numbers match
`estimate_lost_revenue.py`'s own printed summary exactly.
