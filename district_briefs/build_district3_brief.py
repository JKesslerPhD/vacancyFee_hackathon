"""
Build the District 3 brief (print-ready HTML, matches the Vacancy Fee
Project's existing "NO MORE WASTED SPACE" brief brand: red header banner,
dark subheader bar, red section labels, red/cream tables, dark "GET
INVOLVED" CTA box, red footer bar). Portrait, 8.5in x 11in.

Framing: citywide statistics are the headline (6,518 vacant properties
across all 8 districts), with District 3 called out for context (its own
totals, and a highlighted table of its vacant commercial land). Earlier
drafts led with District 3 alone; citywide framing is both more persuasive
(the problem isn't unique to one district) and more defensible (District 3
specifically has an unusually large share of unbuildable marsh/drainage
parcels among its "vacant" count, being mostly Natomas basin geography, so
leading with District-3-only numbers overstated how much of that district's
count is developable land).

Run after revenue_impact/district3_report.py (needs district3_summary.json,
district3_vacant_properties.csv, and the three district3_*.png charts it
produces).

Outputs:
    district_briefs/district3_brief.html -- self-contained (charts + logo
        embedded as base64), 8.5in x 11in portrait, print-ready.
    district_briefs/district3_brief.docx -- same content, plain Word
        formatting (single column, standard styles), meant to be easy to
        edit, not a copy of the HTML's print layout.

To get a PDF from the HTML: open in Chrome/Chromium and print to PDF (or
use Playwright's page.pdf() headlessly), Layout: Portrait, Margins: None,
Background graphics: ON (needed for the red/black/cream brand colors).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIEF_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "revenue_impact" / "figures"
RESULTS_DIR = REPO_ROOT / "revenue_impact" / "results"
LOGO_PATH = Path(
    "/Users/jeffkessler/Library/CloudStorage/GoogleDrive-jeff.kessler@gmail.com/"
    "Other computers/My MacBook Pro/Documents/2025 - 2026/Vacancy Fee/Images/"
    "Vacancy Fee Project Logo.png"
)

RED = "#A6352C"
BLACK = "#1A1A1A"
CREAM = "#F3EAD9"
CREAM_LIGHT = "#FBF7F0"
GRAY = "#5A6068"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def prepare_logo() -> Path | None:
    """Downscale the (1024x1024) source logo for embedding -- full res
    would add roughly 2MB of base64 to the page for a 27px-tall corner mark."""
    if not LOGO_PATH.exists():
        return None
    from PIL import Image
    target = BRIEF_DIR / "logo_small.png"
    im = Image.open(LOGO_PATH)
    im = im.resize((300, int(300 * im.size[1] / im.size[0])), Image.LANCZOS)
    im.save(target)
    return target


def fmt_dollars(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.0f}K"
    return f"${v:,.0f}"


def load_top_commercial(n: int = 4) -> list[dict]:
    """District 3's vacant commercial land, highest tax-uplift potential
    first. Only rows with a real street number -- SITE_ADDR is sometimes
    just a corridor name (e.g. "DEL PASO BLVD") for unaddressed parcels,
    which reads as an error on a printed page rather than what it is.

    No "use" column: the assessor's USE_CODE_STD_DESC_LPS text isn't
    reliable enough at the individual-parcel level to print as fact (see
    the module docstring in district3_report.py -- one parcel flagged this
    way turned out to look like a detached single-family home in a street
    view, not vacant land at all).
    """
    import pandas as pd
    df = pd.read_csv(RESULTS_DIR / "district3_vacant_properties.csv")
    comm = df[df["property_type_group"] == "commercial"]
    has_street_number = comm["SITE_ADDR"].astype(str).str.match(r"^\s*\d")
    comm = comm[has_street_number].drop_duplicates(subset="SITE_ADDR").head(n)
    rows = []
    for _, r in comm.iterrows():
        rows.append({
            "addr": str(r["SITE_ADDR"]).title(),
            "value": fmt_dollars(r["est_market_value"]),
            "uplift": fmt_dollars(r["potential_property_tax_uplift"]) + "/yr",
        })
    return rows


def build_html() -> str:
    summary = json.loads((RESULTS_DIR / "district3_summary.json").read_text())
    top = load_top_commercial()

    logo_b64 = b64(BRIEF_DIR / "logo_small.png") if (BRIEF_DIR / "logo_small.png").exists() else None
    revenue_chart = b64(FIGURES_DIR / "district3_revenue_potential.png")
    duration_chart = b64(FIGURES_DIR / "district3_vacancy_duration.png")
    types_chart = b64(FIGURES_DIR / "district3_property_types.png")

    top_rows = "".join(
        f"""<tr><td>{r['addr']}</td><td class="num">{r['value']}</td><td class="num">{r['uplift']}</td></tr>"""
        for r in top
    )

    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="Vacancy Fee Project logo" '
        f'style="height:27px;width:auto;vertical-align:middle;margin-right:8px;">'
        if logo_b64 else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vacancy Brief: Vacancy Fee Project</title>
<style>
  @page {{ size: 8.5in 11in portrait; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0; width: 8.5in; height: 11in;
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: {BLACK}; background: white;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }}
  .page {{ width: 8.5in; height: 11in; display: flex; flex-direction: column; overflow: hidden; }}

  header.banner {{
    background: {RED}; color: white; padding: 15px 36px 11px;
    text-align: center; flex-shrink: 0;
  }}
  header.banner h1 {{ margin: 0; font-size: 31px; font-weight: 800; letter-spacing: 0.015em; }}
  header.banner .sub {{ margin: 3px 0 0; font-size: 14px; font-weight: 400; opacity: 0.95; }}

  .subbar {{
    background: {BLACK}; color: white; display: flex; justify-content: space-between; align-items: center;
    padding: 6px 36px; font-size: 11px; font-weight: 600; flex-shrink: 0;
  }}
  .subbar .left {{ display: flex; align-items: center; }}

  main {{ flex: 1; padding: 11px 36px 5px; overflow: hidden; display: flex; flex-direction: column; }}

  .label {{ color: {RED}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 1px; }}
  h2 {{ margin: 0 0 4px; font-size: 16.5px; font-weight: 800; line-height: 1.1; }}
  p {{ margin: 0 0 6px; font-size: 10.2px; line-height: 1.35; color: #2a2f34; }}

  .stat-strip {{ display: flex; gap: 10px; margin: 6px 0 8px; }}
  .stat-box {{ flex: 1; background: {CREAM_LIGHT}; border: 1px solid #e2ddd6; border-radius: 7px; padding: 7px 9px; text-align: center; }}
  .stat-box .v {{ font-size: 20px; font-weight: 800; color: {RED}; line-height: 1.05; }}
  .stat-box .l {{ font-size: 9px; color: {GRAY}; margin-top: 3px; line-height: 1.2; }}

  .two-up {{ display: flex; gap: 20px; margin: 2px 0 5px; }}
  .two-up > div {{ flex: 1; min-width: 0; }}

  img.chart {{ width: 100%; max-height: 148px; object-fit: contain; }}
  img.chart.wide {{ max-height: 130px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 9.8px; margin: 2px 0 6px; }}
  table.key-fact td {{ padding: 3px 8px; border-bottom: 1px solid #ece7e0; }}
  table.key-fact td:first-child {{ background: {CREAM}; font-weight: 700; width: 62%; }}
  table th {{ background: {RED}; color: white; text-align: left; padding: 4px 8px; font-size: 9px; text-transform: uppercase; letter-spacing: 0.03em; }}
  table td {{ padding: 3.5px 8px; border-bottom: 1px solid #ece7e0; }}
  table tr:nth-child(even) td {{ background: {CREAM_LIGHT}; }}
  table td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; white-space: nowrap; }}

  .callout-box {{
    background: {CREAM}; border-left: 4px solid {RED}; border-radius: 5px;
    padding: 7px 12px; font-size: 10.3px; font-weight: 700; line-height: 1.3; margin: 3px 0 7px;
  }}

  .cta {{
    background: {BLACK}; color: white; border-radius: 7px; padding: 9px 18px;
    display: flex; justify-content: space-between; align-items: center; margin-top: auto; flex-shrink: 0;
  }}
  .cta .t {{ font-size: 12.5px; font-weight: 800; }}
  .cta .s {{ font-size: 10px; color: #d8d3ca; }}
  .cta .url {{ font-size: 12.5px; font-weight: 800; color: white; }}

  footer.foot {{
    background: {RED}; color: white; text-align: center; font-size: 8.5px;
    padding: 5px 20px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div class="page">

  <header class="banner">
    <h1>NO MORE WASTED SPACE</h1>
    <div class="sub">The Case for a Sacramento Vacancy Tax</div>
  </header>

  <div class="subbar">
    <span class="left">{logo_html}Citywide Briefing, with District 3 Context for Councilmember Karina Talamantes</span>
    <span>VacancyFee.org</span>
  </div>

  <main>
    <div class="label">The Problem</div>
    <h2>Vacancy Is a Choice, Not a Circumstance</h2>
    <p>
      For a lot of property owners, an empty building or lot in Sacramento
      isn't a failure to find a tenant. It's a business model. When land
      appreciates faster than the cost of taxes and interest, sitting on it
      empty can earn more than renting it out ever would. A registration or
      monitoring fee won't change that math: it can catch neglect, but it
      does nothing to the underlying incentive that rewards speculative
      land-banking. A vacancy tax fixes that by putting a real cost on
      doing nothing, while a 90-day grace period and exemptions for active
      construction or bona fide use mean it never touches an owner who is
      actually trying to lease.
    </p>

    <div class="stat-strip">
      <div class="stat-box"><div class="v">{summary['city_n_vacant_properties']:,}</div><div class="l">Vacant/underused<br>properties citywide</div></div>
      <div class="stat-box"><div class="v">{fmt_dollars(summary['city_total_property_tax_uplift'])}/yr</div><div class="l">Property tax<br>uplift potential</div></div>
      <div class="stat-box"><div class="v">{fmt_dollars(summary['city_total_sales_tax'])}/yr</div><div class="l">Lost sales tax<br>revenue</div></div>
    </div>

    <div class="two-up">
      <div>
        <img class="chart wide" src="data:image/png;base64,{types_chart}" alt="Property type breakdown">
        <p style="font-size:9px; color:{GRAY}; margin-top:2px;">
          71% is vacant land zoned residential: unimproved lots or
          structures the assessor rates as having no meaningful
          improvement value. Not occupied homes.
        </p>
      </div>
      <div>
        <img class="chart wide" src="data:image/png;base64,{revenue_chart}" alt="Sales tax lost by council district, District 3 highlighted">
        <p style="font-size:9px; color:{GRAY}; margin-top:2px;">
          District 3 alone accounts for {fmt_dollars(summary['d3_total_sales_tax'])}/yr in
          lost sales tax and {fmt_dollars(summary['d3_total_property_tax_uplift'])}/yr in
          property tax uplift, across {summary['d3_n_vacant_properties']} identified
          vacant or underused properties.
        </p>
      </div>
    </div>

    <div class="label" style="margin-top:2px;">The Pattern</div>
    <h2>These Properties Have Been Sitting for Years</h2>
    <p>
      Of the {summary['city_n_vacant_properties']:,} vacant properties identified citywide, about
      half have a recorded sale date. Among those, the typical property hasn't
      sold in {summary['city_median_years_since_sale']:.0f} years, and
      {summary['city_pct_vacant_10plus_years']*100:.0f}% haven't changed hands in over a decade.
      This is scoped entirely to properties already flagged as vacant, not a claim
      about the city's housing stock broadly.
    </p>
    <img class="chart" src="data:image/png;base64,{duration_chart}" alt="Vacancy duration distribution, vacant properties only">

    <div class="two-up" style="margin-top:6px;">
      <div>
        <div class="label">Precedent</div>
        <h2 style="font-size:13px;">Other California Cities Are Already Acting</h2>
        <table>
          <tr><th>City</th><th>Fee / Tax</th></tr>
          <tr><td>Oakland</td><td>$3K&ndash;$6K/parcel</td></tr>
          <tr><td>Berkeley</td><td>$3K&ndash;$6K/parcel</td></tr>
          <tr><td>Stockton</td><td>$826/yr commercial</td></tr>
          <tr><td>SF (Commercial)</td><td>$250&ndash;$1K/linear ft.</td></tr>
        </table>
      </div>
      <div>
        <div class="label">District 3 Snapshot</div>
        <table class="key-fact">
          <tr><td>Vacant/underused properties</td><td>{summary['d3_n_vacant_properties']}</td></tr>
          <tr><td>Property tax uplift potential</td><td>{fmt_dollars(summary['d3_total_property_tax_uplift'])}/yr</td></tr>
          <tr><td>Lost sales tax revenue</td><td>{fmt_dollars(summary['d3_total_sales_tax'])}/yr</td></tr>
        </table>
      </div>
    </div>

    <div class="label">Vacant Commercial Land in District 3</div>
    <table>
      <tr><th>Address</th><th>Est. value</th><th>Tax uplift</th></tr>
      {top_rows}
    </table>

    <div class="cta">
      <div>
        <div class="t">GET INVOLVED</div>
        <div class="s">Sign up, submit comments, or volunteer</div>
      </div>
      <div class="url">VacancyFee.org</div>
    </div>
  </main>

  <footer class="foot">Vacancy Fee Project &bull; 501(c)(4) &bull; Sacramento, CA &bull; VacancyFee.org &bull; Based on Sacramento County assessor records and parcel-level market value modeling</footer>

</div>
</body>
</html>
"""


def build_docx() -> None:
    """Word version: plain single-column document with Word-native styles
    (headings, tables), not a recreation of the HTML's print layout. Meant
    to be easy to open and edit, not pixel-identical to the PDF."""
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    summary = json.loads((RESULTS_DIR / "district3_summary.json").read_text())
    top = load_top_commercial()

    RED_RGB = RGBColor(0xA6, 0x35, 0x2C)
    BLACK_RGB = RGBColor(0x1A, 0x1A, 0x1A)
    GRAY_RGB = RGBColor(0x5A, 0x60, 0x68)

    def shade_cell(cell, hex_color: str):
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), hex_color)
        cell._tc.get_or_add_tcPr().append(shd)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, m, Inches(0.7))

    title = doc.add_heading("NO MORE WASTED SPACE", level=0)
    title.runs[0].font.color.rgb = RED_RGB
    sub = doc.add_paragraph("The Case for a Sacramento Vacancy Tax")
    sub.runs[0].font.size = Pt(13)
    sub.runs[0].font.color.rgb = GRAY_RGB

    p = doc.add_paragraph()
    r = p.add_run("Citywide Briefing, with District 3 Context for Councilmember Karina Talamantes")
    r.bold = True
    r2 = p.add_run("  |  VacancyFee.org")
    r2.font.color.rgb = GRAY_RGB
    doc.add_paragraph()

    def heading(text, size=14):
        h = doc.add_heading(text, level=2)
        h.runs[0].font.color.rgb = BLACK_RGB
        h.runs[0].font.size = Pt(size)
        return h

    def label(text):
        p = doc.add_paragraph()
        r = p.add_run(text.upper())
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RED_RGB
        return p

    label("The Problem")
    heading("Vacancy Is a Choice, Not a Circumstance")
    doc.add_paragraph(
        "For a lot of property owners, an empty building or lot in Sacramento isn't a "
        "failure to find a tenant. It's a business model. When land appreciates faster "
        "than the cost of taxes and interest, sitting on it empty can earn more than "
        "renting it out ever would. A registration or monitoring fee won't change that "
        "math: it can catch neglect, but it does nothing to the underlying incentive "
        "that rewards speculative land-banking. A vacancy tax fixes that by putting a "
        "real cost on doing nothing, while a 90-day grace period and exemptions for "
        "active construction or bona fide use mean it never touches an owner who is "
        "actually trying to lease."
    )

    label("The Numbers, Citywide")
    stats = doc.add_table(rows=1, cols=3)
    stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = stats.rows[0].cells
    stat_data = [
        (f"{summary['city_n_vacant_properties']:,}", "Vacant/underused properties citywide"),
        (fmt_dollars(summary["city_total_property_tax_uplift"]) + "/yr", "Property tax uplift potential"),
        (fmt_dollars(summary["city_total_sales_tax"]) + "/yr", "Lost sales tax revenue"),
    ]
    for cell, (val, lbl) in zip(hdr, stat_data):
        shade_cell(cell, "FBF7F0")
        p1 = cell.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(val)
        r1.bold = True
        r1.font.size = Pt(16)
        r1.font.color.rgb = RED_RGB
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(lbl)
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = GRAY_RGB
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.add_run(
        "71% of these properties are vacant land zoned residential: unimproved lots or "
        "structures the assessor rates as having no meaningful improvement value. "
    )
    b = p.add_run("Not occupied homes.")
    b.bold = True
    p.add_run(
        " About a fifth are vacant commercial land, sitting on corridors that could "
        "otherwise host local business."
    )
    doc.add_paragraph(
        f"District 3 alone accounts for {fmt_dollars(summary['d3_total_sales_tax'])}/yr in lost "
        f"sales tax and {fmt_dollars(summary['d3_total_property_tax_uplift'])}/yr in property tax "
        f"uplift, across {summary['d3_n_vacant_properties']} identified vacant or underused properties."
    )

    label("The Pattern")
    heading("These Properties Have Been Sitting for Years")
    doc.add_paragraph(
        f"Of the {summary['city_n_vacant_properties']:,} vacant properties identified citywide, "
        "about half have a recorded sale date. Among those, the typical property hasn't sold in "
        f"{summary['city_median_years_since_sale']:.0f} years, and "
        f"{summary['city_pct_vacant_10plus_years']*100:.0f}% haven't changed hands in over a decade. "
        "This is scoped entirely to properties already flagged as vacant, not a claim about the "
        "city's housing stock broadly."
    )

    label("District 3 Snapshot")
    kf = doc.add_table(rows=0, cols=2)
    kf.style = "Light Grid Accent 2"
    kf_rows = [
        ("Vacant/underused properties", str(summary["d3_n_vacant_properties"])),
        ("Property tax uplift potential", fmt_dollars(summary["d3_total_property_tax_uplift"]) + "/yr"),
        ("Lost sales tax revenue", fmt_dollars(summary["d3_total_sales_tax"]) + "/yr"),
    ]
    for k, v in kf_rows:
        row = kf.add_row().cells
        row[0].text = k
        row[0].paragraphs[0].runs[0].bold = True
        row[1].text = v
    doc.add_paragraph()

    label("Vacant Commercial Land in District 3")
    t = doc.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 2"
    hdr_cells = t.rows[0].cells
    for cell, text in zip(hdr_cells, ["Address", "Est. value", "Tax uplift"]):
        shade_cell(cell, "A6352C")
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for r in top:
        row = t.add_row().cells
        row[0].text = r["addr"]
        row[1].text = r["value"]
        row[2].text = r["uplift"]
    doc.add_paragraph()

    label("Precedent")
    heading("Other California Cities Are Already Acting", size=12)
    pr = doc.add_table(rows=1, cols=2)
    pr.style = "Light Grid Accent 2"
    for cell, text in zip(pr.rows[0].cells, ["City", "Fee / Tax"]):
        shade_cell(cell, "A6352C")
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for city, fee in [
        ("Oakland", "$3K–$6K/parcel"),
        ("Berkeley", "$3K–$6K/parcel"),
        ("Stockton", "$826/yr commercial"),
        ("San Francisco (Commercial)", "$250–$1K/linear ft."),
    ]:
        row = pr.add_row().cells
        row[0].text = city
        row[1].text = fee
    doc.add_paragraph()

    cta = doc.add_paragraph()
    cta_run = cta.add_run("GET INVOLVED. Sign up, submit comments, or volunteer at VacancyFee.org")
    cta_run.bold = True
    cta_run.font.size = Pt(12)
    cta_run.font.color.rgb = RED_RGB

    foot = doc.add_paragraph()
    foot_run = foot.add_run(
        "Vacancy Fee Project. 501(c)(4). Sacramento, CA. VacancyFee.org. "
        "Based on Sacramento County assessor records and parcel-level market value modeling."
    )
    foot_run.font.size = Pt(8)
    foot_run.font.color.rgb = GRAY_RGB

    target = BRIEF_DIR / "district3_brief.docx"
    doc.save(target)
    print(f"wrote {target.relative_to(REPO_ROOT)}")


def main() -> None:
    prepare_logo()
    html = build_html()
    target = BRIEF_DIR / "district3_brief.html"
    target.write_text(html)
    print(f"wrote {target.relative_to(REPO_ROOT)} ({target.stat().st_size / 1e3:.0f} KB)")
    build_docx()


if __name__ == "__main__":
    main()
