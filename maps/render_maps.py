"""
Vacancy x 311 — pyQGIS map suite
===============================================================================
Renders a sleek suite of presentation maps over a muted OpenStreetMap basemap
for the vacancy-fee story:

    1. health_safety_311_county   — H&S 311-call density, whole urbanized county
    2. health_safety_311_midtown  — same density zoomed to downtown / midtown
    3. vacant_parcels_county      — coded-vacant + abandoned parcels, county
    4. vacant_parcels_midtown     — same, downtown / midtown
    5. synthesis_county           — H&S density + vacant parcels outlined, county
    6. synthesis_midtown          — H&S density + vacant parcels, midtown
    7. predicted_vacancy_county   — 311-predicted candidate vacancies, county
    8. predicted_vacancy_midtown  — 311-predicted candidate vacancies, midtown
    9. council_districts          — Sacramento city council district boundaries

("Health & safety" / nuisance is the project's reframing of what older work
called "blight" — a term urban-land-use research avoids for its racist
urban-renewal connotations.)

Cartographic notes:
- Basemap opacity is dropped 15% so the OSM detail recedes and the data leads.
- The Sacramento County boundary is drawn on every county-extent map.
- "Zero improvement value" (Tier 2) parcels are EXCLUDED from every map — only
  the coded-vacant (Tier 1) and parking/abandoned (Tier 3) sets are shown.
- Every map carries a proper scale bar and north arrow; chrome (frames, boxes)
  is minimised.
- The 311 "heatmap" is a smoothed density RASTER (numpy/GDAL), not
  QgsHeatmapRenderer, whose auto-scaling is dominated by a 4,400-call geocode
  pile-up that washes the real clusters out.

Run inside the QGIS python environment (Windows):

    "C:\\Program Files\\QGIS 3.38.1\\bin\\python-qgis.bat" maps\\render_maps.py

Inputs:
    maps/data/hs_311.gpkg                  (health & safety 311 points, EPSG:4326)
    maps/data/sacramento_county.geojson    (county boundary; built by prep_layers)
    hackathon_data/vacant_parcels.geojson  (vacant parcels, EPSG:4326)
    data/council_districts/Council_Districts.shp
    maps/data/predicted_vacancies.gpkg     (optional; from predict_vacancy.py)

Outputs: maps/figures/*.png   (+ intermediate density GeoTIFFs in maps/data/)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from osgeo import ogr, gdal, osr

from qgis.core import (
    QgsApplication, QgsProject, QgsRasterLayer, QgsVectorLayer,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsRectangle,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsFillSymbol,
    QgsSingleBandPseudoColorRenderer, QgsColorRampShader, QgsRasterShader,
    QgsBilinearRasterResampler,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat,
    QgsTextBufferSettings,
    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel, QgsLayoutItemLegend,
    QgsLayoutItemScaleBar, QgsLayoutItemPicture, QgsLayoutPoint, QgsLayoutSize,
    QgsLayoutMeasurement, QgsUnitTypes, QgsLayoutExporter,
)
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtCore import Qt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
HS_GPKG = DATA_DIR / "hs_311.gpkg"
PREDICTED_GPKG = DATA_DIR / "predicted_vacancies.gpkg"
COUNTY_GEOJSON = DATA_DIR / "sacramento_county.geojson"
# Council districts reprojected to WGS84 by prep_layers.py — the raw 2226
# shapefile does not reproject reliably onto the 3857 basemap in a layout.
COUNCIL_GEOJSON = DATA_DIR / "council_districts.geojson"
# Prefer the QC-corrected export (public park/rec-district parcels removed,
# see hackathon_data/qc_park_exclusion.py) when it's been generated locally.
VACANT_GEOJSON = PROJECT_ROOT / "hackathon_data" / "vacant_parcels_qc.geojson"
if not VACANT_GEOJSON.exists():
    VACANT_GEOJSON = PROJECT_ROOT / "hackathon_data" / "vacant_parcels.geojson"
OUT_DIR = SCRIPT_DIR / "figures"

# ── Extents (lon/lat, WGS84) ─────────────────────────────────────────────────
EXTENT_COUNTY = (-121.610, 38.160, -121.150, 38.790)   # urbanized Sacramento Co.
EXTENT_MIDTOWN = (-121.505, 38.555, -121.455, 38.590)  # downtown + midtown grid
EXTENT_CITY = (-121.575, 38.430, -121.348, 38.695)     # Sacramento city (districts)

# ── Basemap ──────────────────────────────────────────────────────────────────
OSM_XYZ = ("type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
           "&zmax=19&zmin=0")
BASEMAP_OPACITY = 0.85          # 15% lighter than full — lets the data lead

# ── Sleek palette ────────────────────────────────────────────────────────────
C_TEXT = "#22303C"              # near-black slate for titles/labels
C_MUTED = "#6b7682"             # captions
C_COUNTY = "#2b2d42"            # county boundary
C_DISTRICT = "#3a0ca3"         # district boundary
C_VACANT = "#E5484D"            # Tier 1 coded vacant (refined coral-red)
C_ABANDON = "#0d9488"           # Tier 3 parking / abandoned (teal)
C_PREDICT = "#b5179e"          # 311-predicted candidates (magenta-purple)
C_KNOWN_PREDICT = "#c77dff"    # known-vacant context on the predicted map (lighter purple, same family)

# Tier 2 ("Zero Improvement") is deliberately dropped from every map.
KEPT_TIERS = [
    ("Tier 1: Coded Vacant", C_VACANT, "Coded vacant (land-use 'I')"),
    ("Tier 3: Parking/Abandoned", C_ABANDON, "Parking lot / abandoned"),
]
DROP_TIER2 = "\"vacancy_tier\" <> 'Tier 2: Zero Improvement'"

WEB_MERCATOR = QgsCoordinateReferenceSystem("EPSG:3857")
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


# ── Density raster build (numpy + GDAL) ──────────────────────────────────────

_PTS_CACHE = None


def _load_points():
    """Return (lon, lat) numpy arrays of all health-&-safety 311 points (cached)."""
    global _PTS_CACHE
    if _PTS_CACHE is not None:
        return _PTS_CACHE
    ds = ogr.Open(str(HS_GPKG))
    lyr = ds.GetLayer("hs_311")
    xs, ys = [], []
    for f in lyr:
        g = f.GetGeometryRef()
        if g is not None:
            xs.append(g.GetX())
            ys.append(g.GetY())
    _PTS_CACHE = (np.asarray(xs), np.asarray(ys))
    print(f"  loaded {len(xs):,} health & safety points for density build")
    return _PTS_CACHE


def _smooth(a, passes):
    """Light separable blur (no scipy dependency)."""
    for _ in range(passes):
        b = a.copy()
        b[1:-1, 1:-1] = (
            a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
            + 4.0 * a[1:-1, 1:-1]
        ) / 8.0
        a = b
    return a


def build_density_raster(bbox, cell_deg, smooth_passes, clip_pct, out_tif):
    """Histogram H&S points into a grid, smooth, clip the outlier, write TIF."""
    lon, lat = _load_points()
    minx, miny, maxx, maxy = bbox
    ncols = int(round((maxx - minx) / cell_deg))
    nrows = int(round((maxy - miny) / cell_deg))
    H, _, _ = np.histogram2d(
        lon, lat, bins=[ncols, nrows], range=[[minx, maxx], [miny, maxy]])
    grid = H.T.astype("float32")
    grid = _smooth(grid, smooth_passes)
    nz = grid[grid > 0]
    clip = float(np.percentile(nz, clip_pct)) if nz.size else 1.0
    top = grid[::-1].copy()

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(out_tif), ncols, nrows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([minx, (maxx - minx) / ncols, 0,
                        maxy, 0, -(maxy - miny) / nrows])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(top)
    band.SetNoDataValue(0.0)
    band.FlushCache()
    ds = None
    print(f"  built {out_tif.name}  ({ncols}x{nrows} cells, clip@{clip_pct}pct={clip:.1f})")
    return clip


# ── Layer factories ──────────────────────────────────────────────────────────

def basemap_layer():
    lyr = QgsRasterLayer(OSM_XYZ, "OpenStreetMap", "wms")
    if not lyr.isValid():
        raise RuntimeError("OSM basemap failed to load")
    lyr.setOpacity(BASEMAP_OPACITY)
    return lyr


def density_layer(tif_path, clip, name="Health & safety call density"):
    """Single-band pseudocolour raster — a smooth, refined warm ramp."""
    lyr = QgsRasterLayer(str(tif_path), name)
    if not lyr.isValid():
        raise RuntimeError(f"density raster invalid: {tif_path}")
    lyr.setCrs(WGS84)
    # Faint amber → orange → deep crimson → maroon, with a gentle alpha climb so
    # the quiet areas stay out of the way and the hot cores read cleanly.
    stops = [
        (0.05, QColor(255, 236, 179, 30)),
        (0.22, QColor(255, 193, 84, 110)),
        (0.45, QColor(247, 118, 53, 180)),
        (0.70, QColor(214, 40, 57, 220)),
        (1.00, QColor(140, 18, 60, 240)),
    ]
    items = [QgsColorRampShader.ColorRampItem(frac * clip, col, f"{frac:.2f}")
             for frac, col in stops]
    ramp = QgsColorRampShader(0, clip)
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(items)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    lyr.setRenderer(QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, shader))
    lyr.resampleFilter().setZoomedInResampler(QgsBilinearRasterResampler())
    lyr.resampleFilter().setZoomedOutResampler(QgsBilinearRasterResampler())
    lyr.setOpacity(0.9)
    return lyr


def vacant_layer(outline_only=False, name="Vacant parcels"):
    """Coded-vacant (Tier 1) + parking/abandoned (Tier 3); Tier 2 excluded."""
    lyr = QgsVectorLayer(str(VACANT_GEOJSON), name, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"vacant layer invalid: {VACANT_GEOJSON}")
    if not lyr.crs().isValid():
        lyr.setCrs(WGS84)
    lyr.setSubsetString(DROP_TIER2)
    cats = []
    for tier, color, label in KEPT_TIERS:
        if outline_only:
            props = {"color": "0,0,0,0", "outline_color": color,
                     "outline_width": "0.5", "style": "solid"}
        else:
            # Flat fill, hairline same-hue outline — clean at any zoom.
            props = {"color": color, "outline_style": "no"}
        sym = QgsFillSymbol.createSimple(props)
        cats.append(QgsRendererCategory(tier, sym, label))
    lyr.setRenderer(QgsCategorizedSymbolRenderer("vacancy_tier", cats))
    lyr.setOpacity(1.0 if outline_only else 0.92)
    return lyr


def _single_fill_layer(uri, props, name, provider="ogr", opacity=1.0,
                       subset=None, label_field=None, label_size=15,
                       label_color="#22303C"):
    lyr = QgsVectorLayer(uri, name, provider)
    if not lyr.isValid():
        raise RuntimeError(f"layer invalid: {uri}")
    if not lyr.crs().isValid():
        lyr.setCrs(WGS84)
    if subset:
        lyr.setSubsetString(subset)
    lyr.renderer().setSymbol(QgsFillSymbol.createSimple(props))
    lyr.setOpacity(opacity)
    if label_field:
        pal = QgsPalLayerSettings()
        pal.fieldName = label_field
        pal.enabled = True
        # Centre one horizontal label inside each polygon.
        try:
            pal.placement = QgsPalLayerSettings.Placement.Horizontal
        except AttributeError:
            pal.placement = QgsPalLayerSettings.Horizontal
        fmt = QgsTextFormat()
        fmt.setFont(QFont("Arial", label_size, QFont.Bold))
        fmt.setSize(label_size)
        fmt.setColor(QColor(label_color))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setSize(1.6)
        buf.setColor(QColor("white"))
        fmt.setBuffer(buf)
        pal.setFormat(fmt)
        lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
        lyr.setLabelsEnabled(True)
    return lyr


def vacant_context_layer():
    """Known coded-vacant parcels as a lighter-purple underlay (Tier 2 excluded).

    Deliberately the same hue family as predicted_layer()'s magenta-purple —
    two shades of purple read as "two flavors of vacancy" at a glance, where
    the earlier slate-gray read as "background noise" next to the vivid
    predicted-vacancy magenta.
    """
    return _single_fill_layer(
        str(VACANT_GEOJSON),
        {"color": "199,125,255,110", "outline_style": "no"},
        "Known vacant (coded)", opacity=0.9, subset=DROP_TIER2)


def predicted_layer():
    """311-predicted candidate vacancies (not in the coded set), in magenta."""
    return _single_fill_layer(
        f"{PREDICTED_GPKG}|layername=candidates",
        {"color": "181,23,158,180", "outline_style": "no"},
        "Predicted vacancy (from 311)", opacity=0.95)


def county_layer():
    """Sacramento County boundary — a thin dark hairline, no fill."""
    return _single_fill_layer(
        str(COUNTY_GEOJSON),
        {"color": "0,0,0,0", "outline_color": C_COUNTY, "outline_width": "0.5"},
        "Sacramento County")


_DISTRICT_PASTELS = ["#5b8def", "#34c759", "#ff6b6b", "#ffa94d",
                     "#9775fa", "#22b8cf", "#f06595", "#a9e34b"]


def district_layer():
    """Council districts: a faint distinct fill per district + bold number labels."""
    lyr = QgsVectorLayer(str(COUNCIL_GEOJSON), "Council districts", "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"council layer invalid: {COUNCIL_GEOJSON}")
    if not lyr.crs().isValid():
        lyr.setCrs(WGS84)
    cats = []
    for i in range(1, 9):
        c = QColor(_DISTRICT_PASTELS[(i - 1) % len(_DISTRICT_PASTELS)])
        props = {"color": f"{c.red()},{c.green()},{c.blue()},75",
                 "outline_color": C_DISTRICT, "outline_width": "0.6"}
        cats.append(QgsRendererCategory(
            str(i), QgsFillSymbol.createSimple(props), f"District {i}"))
    lyr.setRenderer(QgsCategorizedSymbolRenderer("DISTNUM", cats))

    pal = QgsPalLayerSettings()
    pal.fieldName = "DISTNUM"
    pal.enabled = True
    try:
        pal.placement = QgsPalLayerSettings.Placement.Horizontal
    except AttributeError:
        pal.placement = QgsPalLayerSettings.Horizontal
    fmt = QgsTextFormat()
    fmt.setFont(QFont("Arial", 20, QFont.Bold))
    fmt.setSize(20)
    fmt.setColor(QColor(C_TEXT))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(2.0)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    pal.setFormat(fmt)
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    lyr.setLabelsEnabled(True)
    return lyr


# ── Layout / render ──────────────────────────────────────────────────────────

def extent_3857(bbox_wgs84):
    ct = QgsCoordinateTransform(WGS84, WEB_MERCATOR, QgsProject.instance())
    minx, miny, maxx, maxy = bbox_wgs84
    return ct.transformBoundingBox(QgsRectangle(minx, miny, maxx, maxy))


def _label(layout, text, x, y, w, h, size, bold=True, color=C_TEXT,
           align=Qt.AlignLeft):
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    f = QFont("Arial")
    f.setPointSizeF(float(size))
    f.setBold(bold)
    item.setFont(f)
    item.setFontColor(QColor(color))
    item.setHAlign(align)
    item.setVAlign(Qt.AlignVCenter)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    item.attemptResize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
    return item


def _add_scalebar(layout, m, x, y, ext_w_m):
    sb = QgsLayoutItemScaleBar(layout)
    sb.setLinkedMap(m)
    sb.setStyle("Single Box")
    # Pick units by map width so midtown (≈4 km) reads in m, county in km.
    units = (QgsUnitTypes.DistanceMeters if ext_w_m < 8000
             else QgsUnitTypes.DistanceKilometers)
    sb.setNumberOfSegments(2)
    sb.setNumberOfSegmentsLeft(0)
    sb.applyDefaultSize(units)                       # sets units AND a round size
    sb.setUnitLabel("m" if units == QgsUnitTypes.DistanceMeters else "km")
    f = QFont("Arial")
    f.setPointSizeF(8.0)
    sb.setFont(f)
    sb.setFillColor(QColor(C_TEXT))
    sb.setFontColor(QColor(C_TEXT))
    try:
        sb.setLineColor(QColor(C_TEXT))
    except AttributeError:
        pass
    sb.setBackgroundEnabled(False)
    layout.addLayoutItem(sb)
    sb.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    return sb


def _add_north(layout, x, y, size=11):
    na = QgsLayoutItemPicture(layout)
    svg = os.path.join(QgsApplication.pkgDataPath(), "svg", "arrows",
                       "NorthArrow_01.svg")  # clean path arrow, no badge box
    na.setPicturePath(svg)
    try:
        na.setSvgFillColor(QColor(C_TEXT))
    except Exception:
        pass
    layout.addLayoutItem(na)
    na.attemptResize(QgsLayoutSize(size, size, QgsUnitTypes.LayoutMillimeters))
    na.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    return na


def render_map(layers, bbox, title, subtitle, out_name, legend_layer=None):
    """Compose a sleek print layout (title, map, legend, scalebar, N) -> PNG."""
    project = QgsProject.instance()
    project.clear()
    project.setCrs(WEB_MERCATOR)
    for lyr in layers:
        project.addMapLayer(lyr, False)

    ext = extent_3857(bbox)
    map_w = 250.0
    aspect = ext.height() / ext.width()
    map_h = max(90.0, min(330.0, map_w * aspect))
    margin, header, footer = 7.0, 22.0, 8.0
    page_w = map_w + 2 * margin
    page_h = header + map_h + footer + margin

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.pageCollection().pages()[0].setPageSize(
        QgsLayoutSize(page_w, page_h, QgsUnitTypes.LayoutMillimeters))

    m = QgsLayoutItemMap(layout)
    m.attemptMove(QgsLayoutPoint(margin, header, QgsUnitTypes.LayoutMillimeters))
    m.attemptResize(QgsLayoutSize(map_w, map_h, QgsUnitTypes.LayoutMillimeters))
    m.setCrs(WEB_MERCATOR)
    m.setLayers(layers)
    m.zoomToExtent(ext)
    m.setBackgroundColor(QColor("#ffffff"))
    # A hairline frame instead of the old heavy border.
    m.setFrameEnabled(True)
    m.setFrameStrokeColor(QColor("#d4d9dd"))
    m.setFrameStrokeWidth(QgsLayoutMeasurement(0.12, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(m)

    _label(layout, title, margin, 4, map_w, 9, size=21, bold=True)
    _label(layout, subtitle, margin, 13.5, map_w, 7, size=11, bold=False,
           color=C_MUTED)

    if legend_layer is not None:
        leg = QgsLayoutItemLegend(layout)
        leg.setTitle("")
        leg.setAutoUpdateModel(False)
        leg.model().rootGroup().clear()
        ll = legend_layer if isinstance(legend_layer, (list, tuple)) else [legend_layer]
        for one in ll:
            leg.model().rootGroup().addLayer(one)
        leg.setLegendFilterByMapEnabled(False)
        leg.setBackgroundColor(QColor(255, 255, 255, 210))
        leg.setBackgroundEnabled(True)
        lf = QFont("Arial")
        lf.setPointSizeF(9.5)
        layout.addLayoutItem(leg)
        leg.attemptMove(QgsLayoutPoint(margin + 2.5, header + 2.5,
                                       QgsUnitTypes.LayoutMillimeters))

    _add_north(layout, margin + map_w - 14, header + 3.5, size=11)
    _add_scalebar(layout, m, margin + 3, header + map_h - 9, ext.width())

    _label(layout, "Source: Sacramento County 311 (SalesForce311) x parcel "
                   "assessor data  ·  Basemap (c) OpenStreetMap contributors  ·  "
                   "vacancyfee.org",
           margin, header + map_h + 1.5, map_w, 6, size=8, bold=False,
           color="#9aa3ab")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exporter = QgsLayoutExporter(layout)
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 150
    res = exporter.exportToImage(str(OUT_DIR / out_name), settings)
    print(f"  {'OK ' if res == QgsLayoutExporter.Success else 'FAIL'} {out_name}")
    project.clear()


# ── Map definitions ──────────────────────────────────────────────────────────

def render_predicted_maps():
    sub = ("Parcels flagged vacant-like by their 311 signal profile, "
           "beyond the coded-vacant set")
    for bbox, place, fname in [
        (EXTENT_COUNTY, "Sacramento County", "predicted_vacancy_county.png"),
        (EXTENT_MIDTOWN, "Downtown & Midtown", "predicted_vacancy_midtown.png"),
    ]:
        pred, ctx = predicted_layer(), vacant_context_layer()
        render_map([pred, ctx, county_layer(), basemap_layer()], bbox,
                   f"311-predicted candidate vacancies — {place}", sub, fname,
                   legend_layer=[pred, ctx])


def render_district_map():
    render_map([district_layer(), basemap_layer()], EXTENT_CITY,
               "Sacramento city council districts",
               "The eight council districts, numbered",
               "council_districts.png")


def main():
    # On macOS, QGIS_PREFIX_PATH must be the .app bundle root itself (e.g.
    # ".../QGIS-LTR.app"), NOT "Contents/MacOS" -- QGIS's bundle-aware plugin
    # path derivation appends "Contents/PlugIns/qgis" unconditionally, so a
    # MacOS-suffixed prefix silently produces a nonexistent doubled-"Contents"
    # plugin path with every non-core provider (including "wms", which the
    # XYZ basemap needs) unregistered and no error raised until layer load.
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], False)
    app.initQgis()
    print("QGIS initialised; building density rasters...")

    county_tif = DATA_DIR / "density_county.tif"
    midtown_tif = DATA_DIR / "density_midtown.tif"
    clip_county = build_density_raster(EXTENT_COUNTY, 0.0006, 6, 96, county_tif)
    clip_mid = build_density_raster(EXTENT_MIDTOWN, 0.00030, 8, 97, midtown_tif)

    print("rendering maps...")
    HS_SUB = ("Density of code-enforcement, dumping & encampment complaints, "
              "2020–2025  ·  darker = more calls")
    VAC_SUB = "Coded-vacant & abandoned parcels (zero-improvement parcels excluded)"

    # 1–2. 311 density
    render_map([density_layer(county_tif, clip_county), county_layer(),
                basemap_layer()], EXTENT_COUNTY,
               "Health & safety 311 call hot spots — Sacramento County",
               HS_SUB, "health_safety_311_county.png")
    render_map([density_layer(midtown_tif, clip_mid), county_layer(),
                basemap_layer()], EXTENT_MIDTOWN,
               "Health & safety 311 call hot spots — Downtown & Midtown",
               HS_SUB, "health_safety_311_midtown.png")

    # 3–4. vacant parcels
    vl = vacant_layer()
    render_map([vl, county_layer(), basemap_layer()], EXTENT_COUNTY,
               "Vacant & underused parcels — Sacramento County",
               VAC_SUB, "vacant_parcels_county.png", legend_layer=vl)
    vl2 = vacant_layer()
    render_map([vl2, county_layer(), basemap_layer()], EXTENT_MIDTOWN,
               "Vacant & underused parcels — Downtown & Midtown",
               VAC_SUB, "vacant_parcels_midtown.png", legend_layer=vl2)

    # 5–6. synthesis
    vl3 = vacant_layer(outline_only=True)
    render_map([vl3, density_layer(county_tif, clip_county), county_layer(),
                basemap_layer()], EXTENT_COUNTY,
               "Vacancy & health-&-safety overlap — Sacramento County",
               "Call density with coded-vacant parcels outlined",
               "synthesis_county.png", legend_layer=vl3)
    vl4 = vacant_layer(outline_only=True)
    render_map([vl4, density_layer(midtown_tif, clip_mid), county_layer(),
                basemap_layer()], EXTENT_MIDTOWN,
               "Vacancy & health-&-safety overlap — Downtown & Midtown",
               "Call density with coded-vacant parcels outlined",
               "synthesis_midtown.png", legend_layer=vl4)

    # 7–8. predicted vacancies
    if PREDICTED_GPKG.exists():
        render_predicted_maps()
    else:
        print(f"  (skipping predicted maps — run predict_vacancy.py to build "
              f"{PREDICTED_GPKG.name})")

    # 9. council districts
    if COUNCIL_GEOJSON.exists():
        render_district_map()

    app.exitQgis()
    print("Done.")


if __name__ == "__main__":
    main()
