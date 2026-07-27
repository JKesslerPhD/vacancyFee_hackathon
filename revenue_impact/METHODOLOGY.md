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

- **Tier 4 ("Predicted (311 Signal)") was including empty residential
  dwellings, which aren't this program's target.** This campaign is about
  vacant commercial/industrial *buildings* and vacant *land* (any zoning) --
  not an empty single-family home or apartment unit. Tier 4 rows are
  appended to `vacant_parcels_qc.csv` by `build_predicted_vacancy_tier.py`
  with no scope check on what kind of property the 311 signal is flagging,
  and thousands of the candidates were ordinary residential dwellings
  (2,623 single-family homes alone). Fixed by excluding candidates whose
  use code is a residential dwelling type
  (`RESIDENTIAL_DWELLING_USE_CODES` in `build_predicted_vacancy_tier.py`),
  dropping 3,990 of 7,083 candidates (56%).

  An earlier version of this fix used the wrong test: it applied
  `qc_vacancy_exclusions.py`'s vacant-*land* allowlist to Tier 4, which
  flagged 7,075 of 7,083 candidates (99.9%) as "miscoded" simply for having
  a building on them at all -- true of nearly every Tier 4 row *by design*
  (Tier 4 exists specifically to find vacant buildings, not vacant land)
  and unrelated to whether that building is a home or a commercial
  property. That version briefly shipped in this repo's history with an
  incorrect claim that it represented a false-positive rate; it did not --
  it measured "does this parcel have a structure," not "is this parcel
  actually vacant" or "is this parcel residential." Corrected here.

- **Tier 1 ("Coded Vacant") wasn't stale-checked.** Tier 1 trusts the
  county's LANDUSE code as of whenever `vacant_parcels.csv` was last built,
  and Sacramento is an actively growing county. Cross-checking every Tier 1
  APN against the *current* LANDUSE code in `data/sac_county_parcel_
  assessors.gpkg` found 4,716 of 19,295 (24%) no longer start with `I`
  (vacant) -- 4,661 of those now show `LU_GENERAL` "Residential", the
  fingerprint of a lot that's since been subdivided and built into homes.
  Fixed with a new check 4 in `qc_vacancy_exclusions.py`. (This check is
  about vacant *land* going stale, unrelated to the residential-dwelling
  issue above -- vacant land zoned residential still counts as vacant
  land regardless of zoning; it's occupied residential *buildings* that
  are out of scope.)

**Important caveat carried forward, not resolved:** there is no
ground-truth occupancy dataset (e.g. a field survey of current tenancy)
to validate Tier 4 predictions against, and `predict_vacancy.py`'s own
"~11% precision proxy" is a self-referential sanity check against the
already-known-vacant set, not an external validation of new candidates --
see that script's docstring. Tier 4 now correctly excludes residential
dwellings, but that says nothing about whether any given flagged
commercial/industrial building is *actually* vacant today. Because Tier 4
candidates skew toward large buildings (offices, department stores,
theaters, hotels), it's currently the dominant contributor to the citywide
totals below despite being the least-validated tier -- see Known
Limitations.

### Current totals

22,454 countywide vacant parcels (22,053 matched a market-value estimate),
8,853 within Sacramento city limits, 5,786 of those capped by
`cap_market_value()` (removing $115.8B in phantom valuation from the city
subset). Citywide headline totals: **$70.0M** potential property tax
uplift, **$295.7M** annual sales tax across all jurisdictions, **$67.6M**
of that as the city's own share. District 3: 654 vacant parcels, $2.6M
potential uplift, $13.6M annual sales tax total / $3.1M city share.

## Sanity cap on est_market_value (applied before everything below)

`ca_property_estimator`'s land-value model produces a small but consequential
share of wildly implausible estimates -- a broken fallback/default constant
firing inside the model for parcels it can't otherwise price cleanly. Left
uncapped, these dominate every downstream total: pre-cap, the largest
parcels in the current city-limited run hit a suspiciously round
**$500 million** ceiling (a "MULTI-TENANT INDUSTRIAL BLDG." assessed at
$1.48M and a "DISTRIBUTION WAREHOUSE" assessed at $884K both price out to
exactly $500M uncapped -- almost certainly the model's own internal clamp,
not two coincidentally-identical market values).

`cap_market_value()` winsorizes `est_market_value` at **$500/sqft** of
`LOT_SIZE_AREA` (falling back to 100× `VAL_ASSD` for parcels missing lot
size). This caps 5,786 of 8,853 vacant parcels within city limits and
removes **$115.8B** in phantom valuation from that subset -- a large swing,
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
totals are computed, not carried through and footnoted. Of the 22,454
countywide vacant parcels, 8,853 (39%) are within city limits and make up
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
- **Tier 4 ("Predicted (311 Signal)") is unvalidated and currently the
  dominant contributor to the citywide totals.** There's no ground-truth
  occupancy dataset available to confirm any individual Tier 4 building is
  actually vacant today — see `build_predicted_vacancy_tier.py`'s
  docstring. Tier 4 correctly excludes residential dwellings (this program
  targets vacant buildings and land, not empty homes), but that's a scope
  filter, not an accuracy filter. If you need a citywide number you can
  defend parcel-by-parcel, filter `parcel_revenue_estimates.csv` to
  `vacancy_tier != "Tier 4: Predicted (311 Signal)"` first — Tiers 1-3 are
  based on the assessor's own recorded land use or improvement value, not
  a model prediction.
- **The $500/sqft sanity cap was calibrated against vacant *land* pricing,
  before Tier 4 added real buildings to the mix.** It divides by
  `LOT_SIZE_AREA` (ground footprint), not total floor area — a multi-story
  building can legitimately be worth far more than $500 per square foot of
  the *lot* it sits on, so this cap may be too aggressive for tall/dense
  commercial buildings and too permissive for single-story ones. It caught
  5,786 of 8,853 city-limited vacant parcels in the current run (up from a
  few hundred before Tier 4 existed) and removed $115.8B in phantom
  valuation — the cap is doing much more work than it was designed for.
  Worth a dedicated pass to calibrate a separate, building-appropriate cap
  (e.g. against improvement value or a $/building-sqft basis) rather than
  reusing the land cap unchanged.
