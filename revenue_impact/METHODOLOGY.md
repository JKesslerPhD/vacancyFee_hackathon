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

## Data refresh (2026-07-26)

Two rounds of fixes landed the same day. Both changed the totals below but
not the underlying methodology.

### Round 1: DUPLICATE_SUMS + SITE_STATE, in ca_property_estimator

Fixed two bugs in `ca_property_estimator/src/data_loader.py`'s
`build_query()` (the query that pulls every California parcel from
Snowflake for scoring):

- **`DUPLICATE_SUMS`.** The query included `AND DUPLICATE_SUMS = FALSE`,
  which silently dropped every parcel flagged as a duplicate-sum record
  from the full pull, not just from the aggregate stats where that flag is
  meant to apply. Multi-unit buildings and condo associations were hit
  hardest. Recovered 45,724 parcels statewide. The clearest local example:
  2417 J St in Sacramento, a highrise apartment building, used to split
  into a dozen-plus fragmentary records with nonsensical assessed values
  (several under $1,000, one at $10) instead of resolving to its two real
  APNs (00700320240000 and 00700320260000).
- **`SITE_STATE`.** The query filtered `WHERE SITE_STATE = 'CA'`, but
  1,253,699 genuine California parcels statewide (~11% of the table) have
  `SITE_STATE` NULL despite every one of them having a California
  `FIPS_CODE` (state prefix `06`) -- confirmed by cross-checking
  `COUNTYNAME`/`FIPS_CODE` for all of them. A handful of rows even had
  `SITE_STATE = 'OR'` while geocoded in Modoc County, CA -- a straight
  data-entry error. Switched the filter to `LEFT(FIPS_CODE, 2) = '06'`,
  a strict superset (every `SITE_STATE = 'CA'` row already has that FIPS
  prefix, zero exceptions). Recovered 1,253,721 parcels statewide
  (11,482,901 rows total after both fixes, up from 10,229,180).

### Round 2: two gaps in the vacant-parcel classification itself

Recovering all those parcels meant `est_market_value` successfully matched
to thousands of Sacramento parcels that had never had a value joined to
them before -- which surfaced two pre-existing gaps in
`hackathon_data/qc_vacancy_exclusions.py` and
`hackathon_data/build_predicted_vacancy_tier.py` that had been silently
contributing $0 to every total until parcels actually had a dollar value to
contribute:

- **Tier 4 ("Predicted (311 Signal)") never went through the use-code
  allowlist.** Tier 4 rows are appended to `vacant_parcels_qc.csv` by
  `build_predicted_vacancy_tier.py` *after* `qc_vacancy_exclusions.py`
  runs, so they never passed through that script's existing
  miscoded-use-code check at all, despite being a noisier, model-predicted
  signal that arguably needs *more* scrutiny than the coded tiers, not
  less. Of 7,083 Tier 4 candidates, 7,075 (99.9%) had a use code implying a
  real, occupied structure -- department stores, high-rise apartments,
  offices, a theater -- because large, busy, occupied buildings generate
  plenty of 311 calls for reasons that have nothing to do with vacancy.
  This one gap accounted for 77.5% of the citywide sales-tax estimate and
  53.2% of the property-tax uplift estimate in an intermediate run. Fixed
  by applying the same allowlist `qc_vacancy_exclusions.py` already trusts
  for Tier 2/3, inside `build_predicted_vacancy_tier.py`, before appending.
- **Tier 1 ("Coded Vacant") wasn't stale-checked.** Tier 1 trusts the
  county's LANDUSE code as of whenever `vacant_parcels.csv` was last built,
  and Sacramento is an actively growing county. Cross-checking every Tier 1
  APN against the *current* LANDUSE code in `data/sac_county_parcel_
  assessors.gpkg` found 4,716 of 19,295 (24%) no longer start with `I`
  (vacant) -- 4,661 of those now show `LU_GENERAL` "Residential", the
  fingerprint of a lot that's since been subdivided and built into homes.
  Smaller impact than the Tier 4 fix ($2.99M of citywide sales tax, $623K
  of property-tax uplift, pre-fix) but a real, verified staleness gap.
  Fixed with a new check 4 in `qc_vacancy_exclusions.py`.

### Current totals

19,369 countywide vacant parcels (19,368 matched a market-value estimate),
5,862 within Sacramento city limits, 1,154 of those capped by
`cap_market_value()` (removing $7.9B in phantom valuation from the city
subset). Citywide headline totals: **$39.2M** potential property tax
uplift, **$71.0M** annual sales tax across all jurisdictions, **$16.2M** of
that as the city's own share. District 3: 478 vacant parcels, $1.7M
potential uplift, $3.8M annual sales tax total / $857K city share.

## Sanity cap on est_market_value (applied before everything below)

`ca_property_estimator`'s land-value model produces a small but consequential
share of wildly implausible estimates -- a broken fallback/default constant
firing inside the model for parcels it can't otherwise price cleanly. Left
uncapped, these dominate every downstream total: pre-cap, the single
largest parcel in the current city-limited run (a ~1-acre
"INDUSTRIAL-VACANT LAND" lot assessed at $148,960) prices out to
**$176.7 million**, a ~$4,088/sqft rate against a citywide median closer to
$100/sqft.

`cap_market_value()` winsorizes `est_market_value` at **$500/sqft** of
`LOT_SIZE_AREA` (falling back to 100× `VAL_ASSD` for parcels missing lot
size). This caps 1,154 of 5,862 vacant parcels within city limits and
removes **$7.9B** in phantom valuation from that subset -- a large swing,
but in the direction of *removing* a fabrication, not introducing one.
`est_market_value_uncapped` is preserved in `parcel_revenue_estimates.csv`
for anyone who wants to audit exactly which parcels were capped and by how
much.

**Marsh/drainage parcels get a tighter cap.** $500/sqft assumes buildable
urban land, which "WASTE LAND, MARSH, SWAMP, SUBMERGED-VACANT LAND"
parcels explicitly aren't (that's what the use code means, and it's a
legitimate, unedited allowlist entry in `qc_vacancy_exclusions.py`, not a
miscoded structure). A large lot size times $500/sqft produces a huge
ceiling regardless of whether the land is a retention basin, so these
1,075 citywide parcels (median assessed/market ratio 72x, worst case
203,000x) fall back to the assessed-value-multiple cap unconditionally,
same as parcels missing `LOT_SIZE_AREA`. Found via a real example flagged
by a campaign volunteer: 3497 San Juan Rd, a Natomas drainage/detention
parcel assessed at $69, was landing at a modeled $1.05M before this fix.
`NON_BUILDABLE_USE_CODES` in `estimate_lost_revenue.py` is the one-item
set this currently checks; District 3 (mostly Natomas basin geography)
is disproportionately affected, at 188 of its 486 flagged vacant
parcels (39%) being this use code, against 16% citywide.

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

### Which parcels count as commercial-eligible

A parcel counts if **either** the assessor's `USE_CODE_STD_DESC_LPS` says
commercial/retail, **or** its `ZONING` is a commercial base code — a union,
not an intersection. This wasn't the original design: it started as
use-code-only, which turned out to badly undercount. A by-hand audit
(prompted by "is this based on zoning?") found 587–754 parcels zoned
`C-1`/`C-2`/`C-3`/`C-4`/`GC`/`SC`/`LC`/etc. but coded `RESIDENTIAL-VACANT
LAND` or `INDUSTRIAL-VACANT LAND` by the assessor — land that's legally
buildable as commercial today, sitting vacant, just not classified that way
in the tax roll. Adding the zoning signal raised the citywide sales-tax
total from $36.2M to **$50.4M/year** (+39%) and roughly doubled District 4's
commercial-eligible count (189 → 355) — Sacramento's downtown/midtown core
has a lot of commercially-zoned vacant lots the use-code alone was missing.

`ZONING` is messy free text (hundreds of raw variants, PUD/SPD/overlay
suffixes, occasional multi-zone entries), so the zoning check only matches a
conservative, verifiable base-code allowlist: `C-1` through `C-4`, `GC`
(general commercial), `SC` (shopping/service commercial), `LC` (limited
commercial), `CC` (community commercial), `BP` (business park), `OB` (office
building), and the mixed-use family (`MU`, `CMU`, `OPMU`, `OIMU`, `RMU`,
`DMU`, `VCMU` — mixed-use zoning explicitly permits ground-floor commercial).
Ambiguous 2-letter codes that couldn't be confidently verified from the data
alone (`SPA`, `HC`, `MP`, `DC`, `TC`, `AC`, ...) are deliberately left out —
undercounting a few is preferable to guessing wrong on codes whose meaning
isn't certain. See `is_zoned_commercial()` in `estimate_lost_revenue.py`
for the exact pattern, and `commercial_basis` in
`parcel_revenue_estimates.csv` (`use_code_only` / `zoning_only` /
`use_code_and_zoning`) to see which signal caught which parcel.

Excluded entirely regardless of zoning: vacant residential, industrial,
waste/marsh, and — per direct instruction — parking-lot parcels (see
`qc_vacancy_exclusions.py`), because a still-operating parking lot may
already generate some untaxed cash income, but converting it to a
sales-tax-generating business isn't a given the way it is for vacant,
commercially-zoned land.

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
totals are computed, not carried through and footnoted. Of the 19,369
countywide vacant parcels, 5,862 (30%) are within city limits and make up
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
