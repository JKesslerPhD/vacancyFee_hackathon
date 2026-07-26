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

The parcel cache and every dollar figure in this file were rebuilt after
fixing a bug in `ca_property_estimator/src/data_loader.py`'s `build_query()`:
the per-parcel scoring query included `AND DUPLICATE_SUMS = FALSE`, which
silently dropped every parcel flagged as a duplicate-sum record from the
full statewide pull, not just from the aggregate stats where that flag is
meant to apply. Multi-unit buildings and condo associations were hit
hardest. The clearest local example: 2417 J St in Sacramento, a highrise
apartment building, used to split into a dozen-plus fragmentary records
with nonsensical assessed values (several under $1,000, one at $10) instead
of resolving to its two real APNs. The fixed query recovered 45,724
previously-dropped parcels statewide (10,229,180 rows total), and 2417 J St
now resolves cleanly to APNs 00700320240000 and 00700320260000.

This changed the totals below but not the underlying methodology. Current
run: 24,077 countywide vacant parcels (23,143 matched a market-value
estimate), 6,518 within Sacramento city limits, 1,487 of those capped by
`cap_market_value()` (removing $6.9B in phantom valuation from the city
subset). Citywide headline totals: **$34.8M** potential property tax
uplift, **$63.5M** annual sales tax across all jurisdictions, **$14.5M** of
that as the city's own share. District 3: 486 vacant parcels, $1.7M
potential uplift, $4.6M annual sales tax total / $1.05M city share.

The specific illustrative numbers in the "Sanity cap" section right below
(the $7,107.14/sqft cluster, the $276M parcel, the 2,702-parcel count) were
measured on the pre-fix run and haven't been individually re-checked
against the new cache. The capping mechanism they describe is unchanged
and still active — a spot check on the current city subset shows the same
pattern (an industrial-vacant parcel modeling out to $158M pre-cap, cut to
$21.6M), just not re-verified figure by figure.

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
totals are computed, not carried through and footnoted. Of the 24,077
countywide vacant parcels, 6,518 (27%) are within city limits and make up
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
