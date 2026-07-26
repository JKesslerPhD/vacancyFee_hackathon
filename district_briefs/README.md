# `district_briefs/` — printable one-pagers for individual council districts

Print-ready briefs matching the Vacancy Fee Project's existing "NO MORE
WASTED SPACE" one-pager brand (red header banner / black subheader bar /
two-column body with red section labels and red-cream tables / dark "GET
INVOLVED" box / red footer bar — see the org's existing
`VacancyFee_CouncilMeetingMarch_OnePager.pdf` and
`Vacancy Fee One Page Handout.pdf` for the reference design).

- **`district3_brief.html`** / **`district3_brief.pdf`** — District 3
  (Councilmember Karina Talamantes). 486 vacant/underused properties,
  annual revenue potential (lost sales tax shown as one headline number,
  not split into the much smaller city general-fund cut), vacancy-duration
  distribution (scoped to the 486 identified vacant properties only, never
  the district's full parcel roll), median stats, and the four
  highest-value vacant parcels with real street-numbered addresses (a few
  of District 3's vacant parcels only have a corridor name on file, e.g.
  "DEL PASO BLVD" with no number — those are filtered out of the table so
  it doesn't look like a data error). All pulled from
  `revenue_impact/results/district3_summary.json` and
  `district3_vacant_properties.csv`. **8.5in x 11in portrait**, one page.
- **`district3_brief.docx`** — same content, Word-native formatting
  (single column, standard heading/table styles, no attempt to recreate
  the HTML's print layout) so it's easy to open and edit directly.
- **`build_district3_brief.py`** — regenerates the HTML *and* the docx from
  the current data + charts. Edit *this*, not the HTML/docx directly —
  both are overwritten on every run.
- **`logo_small.png`** — a 300px-wide downscaled copy of the org's logo
  (`Vacancy Fee Project Logo.png` in the shared Google Drive, 1024x1024
  originally), regenerated automatically by the build script so the page
  doesn't carry ~2MB of unnecessary base64.

Note on the "vacant, zoned-residential land" category: this bucket
(`vacant_land_residential` in `property_type_group()`, see
`revenue_impact/estimate_lost_revenue.py`) is unimproved land zoned
residential, or structures the assessor rates as having no meaningful
improvement value — **not occupied homes**. The brief (and the interactive
map's tooltips) spell this out explicitly; earlier drafts just said
"vacant land / residential," which reads as if lived-in houses were being
counted as vacant.

## Rebuilding

```bash
# 1. revenue_impact/results/parcel_revenue_estimates.csv must exist
python revenue_impact/estimate_lost_revenue.py

# 2. district3_summary.json, district3_vacant_properties.csv, and the
#    three district3_*.png charts in revenue_impact/figures/
python revenue_impact/district3_report.py

# 3. regenerates district_briefs/district3_brief.html AND .docx
#    (HTML is self-contained -- charts + logo embedded as base64;
#    requires python-docx: pip install python-docx)
python district_briefs/build_district3_brief.py
```

To (re)export the PDF: open `district3_brief.html` in Chrome and print to
PDF — Layout: **Portrait**, Paper size: Letter, Margins: None, **Background
graphics: ON** (required for the red/black/cream brand colors to print;
without it the page prints as plain black-on-white text). Or headlessly,
with Playwright:

```js
await page.goto('file://.../district3_brief.html');
await page.pdf({ path: 'district3_brief.pdf', width: '8.5in', height: '11in',
                  printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 } });
```

If you change the CSS, double-check nothing silently overflows: `main` uses
`overflow: hidden` to force a hard one-page limit, so content that's too
tall gets invisibly clipped rather than flowing to a second page. Verify
with `main.scrollHeight` vs `main.clientHeight` in a headless browser (they
must be equal) before trusting a screenshot alone — and note that CSS `in`
units render at 96px/in regardless of viewport pixel density, so a
Playwright viewport for QA screenshots should be sized `816x1056` (8.5x11in
at 96dpi), not scaled up, or you'll see blank canvas around the actual page
and misdiagnose it as overflow.

## Extending to other districts

`build_district3_brief.py` is scoped to District 3 (hardcoded
`DISTRICT_LABEL` in `district3_report.py`, hardcoded copy in the HTML
template). To build another district's brief, the cleanest path is
parameterizing both scripts by district rather than copy-pasting — swap
`DISTRICT_LABEL`, regenerate the summary/charts under a
`district{N}_*` naming scheme, and template the HTML's district-specific
copy (title, councilmember name, narrative paragraphs) instead of the
current hardcoded strings.
