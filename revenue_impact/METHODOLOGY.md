# Unrealized Tax Base — Methodology

`estimate_lost_revenue.py` puts a dollar figure on vacant parcels per council
district, split into two categories that behave completely differently and
should never be added together without saying so:

- **Property tax uplift — contingent on sale/reassessment.** Under Prop 13, a
  parcel's assessed value doesn't move to market value just because it's
  vacant or occupied; it only resets on a sale, transfer, or new
  construction. So this is *not* revenue the city is losing this year — it's
  the tax base that would be unlocked if these parcels changed hands or were
  developed. Reported as a one-time potential uplift, not an annual figure.
- **Sales tax — a genuine recurring annual loss**, for the subset of vacant
  parcels zoned/used commercial or retail. Every year a commercially-viable
  lot sits empty instead of housing a business, that's real foregone taxable
  sales, no reassessment trigger required.

## Sanity cap on est_market_value (applied before everything below)

`ca_property_estimator`'s land-value model produces a small but consequential
share of wildly implausible estimates. The clearest evidence: **1,662
parcels — scattered across residential-vacant, industrial-vacant, and
waste/marsh land, with no relationship to each other — all price out to the
literal same rate, $7,107.14/sqft**, agreeing to 4-5 decimal places. That
isn't market variation; it's a broken fallback/default constant firing
inside the model for parcels it can't otherwise price. Left uncapped, these
dominate every downstream total: pre-cap, the single largest parcel (a
<1-acre "COMMERCIAL-VACANT LAND" lot assessed at $405K) priced out to
**$276 million**, 67% of its entire council district's sales-tax estimate on
its own, and the top 10 of ~900 commercial-eligible parcels citywide made up
56% of the citywide sales-tax total.

`cap_market_value()` winsorizes `est_market_value` at **$500/sqft** of
`LOT_SIZE_AREA` (falling back to 100× `VAL_ASSD` for the ~2.6% of parcels
missing lot size). $500/sqft sits comfortably above the 90th percentile of
the model's own non-broken output (~$105–270/sqft citywide, depending on
exactly where you slice it) and just as comfortably below the $7,107/sqft
cluster — there's a clean, empty gap in the data between roughly $270 and
$7,000/sqft, so the exact cap value isn't sensitive within that range. This
capped 2,702 of ~27,000 vacant parcels citywide and removed **$115B** in
phantom valuation, dropping the citywide headline totals roughly 7-8x (from
~$301M to ~$39M property tax uplift; ~$258M to ~$38M annual sales tax) —
a large swing, but in the direction of *removing* a fabrication, not
introducing one. `est_market_value_uncapped` is preserved in
`parcel_revenue_estimates.csv` for anyone who wants to audit exactly which
parcels were capped and by how much.

## Property tax uplift

```
prop13_gap = max(est_market_value - VAL_ASSD, 0)          # from ca_property_estimator
potential_property_tax_uplift = prop13_gap * 0.011         # 1.1% effective combined rate
```

**Rate: 1.1%.** California's Prop 13 base rate is a flat 1% of assessed
value; Sacramento County parcels layer voter-approved school bonds, city
bonds, and special-district assessments on top, putting most parcels'
effective combined rate in a commonly cited **1.05%–1.3%** range. 1.1% is
the low-middle of that range, which — given the estimator's own noted
tendency toward outlier low values on some vacant parcels (see PR discussion
in the parent commit) — keeps this a conservative, not inflated, estimate.

This is the **total** rate across every taxing entity in the tax rate area
(county, city, schools, special districts) — not the city's share alone.
The city's specific slice of the 1% base levy is set per Tax Rate Area under
Revenue & Taxation Code §96.5 (AB 8, 1979) and isn't published as a single
countywide percentage; getting it precisely would require Sacramento County
Auditor-Controller TRA-level data this repo doesn't have. For scale, the
city's own FY budget materials show property tax as ~29% of General/Measure U
Fund revenue — useful context, not a substitute for a real TRA-level split.

## Sales tax (recurring)

Only computed for parcels whose `USE_CODE_STD_DESC_LPS` contains
"COMMERCIAL" or "RETAIL" (~2,100 of the ~28,400 vacant parcels) — vacant
residential, industrial, waste/marsh, and parking-lot parcels are excluded
because they either can't legally house retail/commercial sales activity or
the estimate would be too speculative to defend (e.g. a still-operating
parking lot may already generate some untaxed cash income, but converting it
to a sales-tax-generating business isn't a given the way it is for
commercial-zoned vacant land).

```
imputed_annual_rent = est_market_value * 0.065              # cap rate
estimated_annual_business_revenue = imputed_annual_rent / 0.08   # occupancy cost ratio
estimated_annual_sales_tax_total = estimated_annual_business_revenue * 0.0875
estimated_annual_sales_tax_city  = estimated_annual_business_revenue * 0.02
```

**Cap rate: 6.5%.** Sacramento retail cap rates were reported around
6.45–6.65% in Q1 2025 for neighborhood/strip retail, with asking cap rates
near 6% and sold cap rates higher (~8.9%) more recently — 6.5% sits in that
band and converts an estimated market value into a plausible annual NNN rent
a landlord could charge if the lot were built out and leased.

**Occupancy cost ratio: 8%.** Commercial real estate underwriting commonly
targets rent (occupancy cost) at 5–10% of a retail tenant's gross sales, with
6–8% cited as "healthy" for retail/QSR specifically. 8% is the middle of
that band; dividing imputed rent by it backs into a plausible gross-sales
figure for the business that would occupy the space.

**Sales tax rate: 8.75% combined / 2% city general fund.** The minimum
combined rate in Sacramento is 8.75% (6% state + 0.25% county + 1% city
Bradley-Burns + 1.5% voter-approved special district, including transit and
other countywide measures). Of that, exactly **2 percentage points is
Sacramento-city-specific and flows to the city's General/Measure U Fund**:
the 1% Bradley-Burns local rate plus the 1% Measure U tax (per
cityofsacramento.gov's own sales-tax and Measure U pages) — a City
Auditor's Baseline that we cite directly, unlike the property-tax city-share
figure above.

## District join — City of Sacramento only

Parcels are matched to a council district by point-in-polygon on
`LATITUDE`/`LONGITUDE` against `maps/data/council_districts.geojson` (the 8
current Sacramento city council districts). This module is scoped to the
**City of Sacramento only**: a parcel that doesn't fall inside any of the 8
district polygons — unincorporated county land, or another incorporated city
in the countywide vacant-parcel set (Elk Grove, Folsom, Citrus Heights,
Rancho Cordova, Galt) — is outside city limits and is dropped before any
totals are computed, not carried through and footnoted. Of the 28,426
countywide vacant parcels, 8,489 (30%) are within city limits and make up
this module's entire output.

## Known limitations

- This is order-of-magnitude estimation for advocacy/planning use, not an
  appraisal. Every rate above is a documented, sourced default, not a
  parcel-specific fact — see the constants at the top of
  `estimate_lost_revenue.py` to substitute your own assumptions.
- `est_market_value` inherits `ca_property_estimator`'s known issue with a
  handful of implausibly *low* vacant-land estimates too (see the parent
  PR's README note) — the opposite direction from the $7,107/sqft cluster
  capped above, and not something a ceiling can fix. Those parcels will
  understate both the property-tax gap and imputed rent for the sales-tax
  estimate.
- The commercial/retail filter is a text match on the assessor's own
  `USE_CODE_STD_DESC_LPS`, not a legal zoning determination — parcels with
  ambiguous or missing use codes fall out of the sales-tax estimate even if
  a real project would qualify.
