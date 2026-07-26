# `results/` — public-facing pages

Two static pages, both embeddable via `<iframe>`:

- **`index.html`** — "The Vacant Equity Gap." Headline figures + interactive
  parcel map. Built by `build_map_data.py`. Needs to be hosted (not opened
  via `file://`) — it `fetch()`es its data files from `map_data/`, which
  Chrome/Safari block from a `file://` page under CORS.
- **`vacancy_explorer.html`** — "Potential Benefits of a Vacancy Fee": a
  property-count choropleth over the City of Sacramento's official 129
  Neighborhoods + council district outlines + revenue overlay, City of
  Sacramento only. Built by `build_vacancy_explorer_data.py` from
  `vacancy_explorer_template.html` **with neighborhoods + districts +
  citywide totals inlined** — so this opens by double-clicking it, no
  server needed, in addition to being hostable/embeddable as-is. Named
  neighborhoods replaced an earlier two-tier Census block-group/block
  design: 129 named, publicly-recognized areas (matching
  [the city's reference map](https://www.cityofsacramento.gov/content/dam/portal/it/gis/Maps/Neighborhoods_E.pdf))
  beat anonymous Census geography, and at that feature count there's no
  need for a separate zoomed-in lazy-loaded layer -- it all inlines
  directly. Edit the *template*, not `vacancy_explorer.html` directly —
  it's regenerated (and overwritten) on every build.

## Rebuilding `vacancy_explorer.html`

```bash
# 1. hackathon_data/vacant_parcels_qc.csv must exist (hackathon_data/qc_vacancy_exclusions.py)
# 2. ca_property_estimator/results/parcels_market_value_estimated.csv must exist --
#    ca_property_estimator is an external project (CARB AB 2446), not part of this
#    repo; get its output CSV from a teammate and place it at that path locally
# 3. revenue_impact/results/{parcel_revenue_estimates,district_revenue_summary}.csv
python revenue_impact/estimate_lost_revenue.py

# 4. reads maps/data/sacramento_neighborhoods_raw.geojson (City of Sacramento's
#    official Neighborhoods layer -- see NEIGHBORHOODS_GEOJSON in the script
#    for where to re-fetch it if it's ever missing), clips to city limits,
#    aggregates onto neighborhoods + districts, writes map_data/*.json,
#    and regenerates vacancy_explorer.html with neighborhoods/districts inlined
python results/build_vacancy_explorer_data.py
```

`map_data/neighborhoods_vacancy.json` and `map_data/council_districts_revenue.json`
are also written standalone and tracked (small, a few hundred KB each) — not
used by `vacancy_explorer.html` itself (which has its own inlined copy) but
handy for debugging or reuse elsewhere without re-running the full build.

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
