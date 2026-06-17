import io
import os
import urllib.request
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def setup_fonts():
    temp_dir = tempfile.gettempdir()
    font_reg_path = os.path.join(temp_dir, "Roboto-Regular.ttf")
    font_bold_path = os.path.join(temp_dir, "Roboto-Bold.ttf")

    url_reg = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"
    url_bold = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Bold.ttf"

    if not os.path.exists(font_reg_path):
        urllib.request.urlretrieve(url_reg, font_reg_path)
    if not os.path.exists(font_bold_path):
        urllib.request.urlretrieve(url_bold, font_bold_path)

    pdfmetrics.registerFont(TTFont('CustomFont', font_reg_path))
    pdfmetrics.registerFont(TTFont('CustomFont-Bold', font_bold_path))

setup_fonts()


PRIMARY    = colors.HexColor("#1a73e8")
SUCCESS    = colors.HexColor("#28a745")
WARNING    = colors.HexColor("#ffc107")
DANGER     = colors.HexColor("#dc3545")
LIGHT_GRAY = colors.HexColor("#f8f9fa")
MID_GRAY   = colors.HexColor("#dee2e6")
DARK_GRAY  = colors.HexColor("#495057")
WHITE      = colors.white
BLACK      = colors.black

RISK_COLORS = {
    "Low":     SUCCESS,
    "Medium":  WARNING,
    "High":    DANGER,
    "Unknown": DARK_GRAY
}


def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title",
            fontSize=24,
            fontName="CustomFont-Bold",
            textColor=PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=11,
            fontName="CustomFont",
            textColor=DARK_GRAY,
            alignment=TA_CENTER,
            spaceAfter=2
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=13,
            fontName="CustomFont-Bold",
            textColor=PRIMARY,
            spaceBefore=14,
            spaceAfter=6
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10,
            fontName="CustomFont",
            textColor=BLACK,
            spaceAfter=4,
            leading=16
        ),
        "risk_label": ParagraphStyle(
            "risk_label",
            fontSize=18,
            fontName="CustomFont-Bold",
            alignment=TA_CENTER
        ),
        "footer": ParagraphStyle(
            "footer",
            fontSize=8,
            fontName="CustomFont",
            textColor=DARK_GRAY,
            alignment=TA_CENTER
        ),
        "suggestion": ParagraphStyle(
            "suggestion",
            fontSize=10,
            fontName="CustomFont",
            textColor=BLACK,
            spaceAfter=6,
            leading=15,
            leftIndent=10
        )
    }
    return custom


def build_report(risk_report: dict, suggestions: str, categorized_df_records: list) -> bytes:
    import pandas as pd
    df = pd.DataFrame(categorized_df_records)

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = get_styles()
    story  = []
    r      = risk_report
    W      = A4[0] - 4*cm


    story.append(Paragraph("FinSight AI", styles["title"]))
    story.append(Paragraph("Personal Finance and Expenditure Report", styles["subtitle"]))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        styles["subtitle"]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width=W, thickness=1.5, color=PRIMARY))
    story.append(Spacer(1, 0.4*cm))


    story.append(Paragraph("Financial Summary", styles["section"]))

    summary_data = [
        ["Monthly Income", "Total Spent", "Savings Left", "Transactions"],
        [
            f"₹ {r['monthly_income']:,.0f}",
            f"₹ {r['total_spent']:,.2f}",
            f"₹ {r['savings_potential']:,.2f}",
            str(r["transaction_count"])
        ]
    ]
    summary_table = Table(summary_data, colWidths=[W/4]*4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "CustomFont-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 10),
        ("ALIGN",         (0, 0), (-1,-1), "CENTER"),
        ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
        ("FONTNAME",      (0, 1), (-1, 1), "CustomFont-Bold"),
        ("FONTSIZE",      (0, 1), (-1, 1), 13),
        ("BACKGROUND",    (0, 1), (-1, 1), LIGHT_GRAY),
        ("GRID",          (0, 0), (-1,-1), 0.5, MID_GRAY),
        ("ROWHEIGHT",     (0, 0), (-1,-1), 28),
        ("TOPPADDING",    (0, 0), (-1,-1), 8),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4*cm))


    story.append(Paragraph("Risk Assessment", styles["section"]))

    risk_level = r.get("risk_level", "Unknown")
    risk_color = RISK_COLORS.get(risk_level, DARK_GRAY)

    risk_data = [[
        Paragraph(
            f"{risk_level} Risk",
            ParagraphStyle(
                "rl",
                fontSize=18,
                fontName="CustomFont-Bold",
                textColor=WHITE,
                alignment=TA_CENTER
            )
        ),
        Paragraph(
            f"{r['risk_score']}% of income spent<br/>Saveable excess: ₹ {r['total_saveable']:,.2f}",
            ParagraphStyle(
                "rd",
                fontSize=11,
                fontName="CustomFont",
                textColor=WHITE,
                alignment=TA_CENTER,
                leading=18
            )
        )
    ]]
    risk_table = Table(risk_data, colWidths=[W*0.35, W*0.65])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,-1), risk_color),
        ("ALIGN",         (0, 0), (-1,-1), "CENTER"),
        ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
        ("ROWHEIGHT",     (0, 0), (-1,-1), 50),
        ("TOPPADDING",    (0, 0), (-1,-1), 10),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 10),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.4*cm))


    story.append(Paragraph("Category Breakdown", styles["section"]))


    CATEGORY_LIMITS = {} 

    cat_header = ["Category", "Amount Spent", "% of Income", "Limit %", "Status"]
    cat_rows   = [cat_header]

    for cat in r.get("category_summary", []):
        limit_pct = CATEGORY_LIMITS.get(cat["category"], 0.15) * 100
        over      = cat["percentage"] > limit_pct
        status    = "Over" if over else "OK"
        cat_rows.append([
            cat["category"],
            f"₹ {cat['total']:,.2f}",
            f"{cat['percentage']}%",
            f"{limit_pct:.0f}%",
            status
        ])

    cat_table = Table(cat_rows, colWidths=[W*0.25, W*0.22, W*0.18, W*0.15, W*0.20])
    cat_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "CustomFont-Bold"),
        ("FONTSIZE",      (0, 0), (-1,-1), 9),
        ("ALIGN",         (1, 0), (-1,-1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
        ("GRID",          (0, 0), (-1,-1), 0.5, MID_GRAY),
        ("ROWHEIGHT",     (0, 0), (-1,-1), 22),
        ("TOPPADDING",    (0, 0), (-1,-1), 5),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 5),
        ("FONTNAME",      (0, 1), (-1,-1), "CustomFont") 
    ]

    for i, cat in enumerate(r.get("category_summary", []), start=1):
        limit_pct = CATEGORY_LIMITS.get(cat["category"], 0.15) * 100
        if cat["percentage"] > limit_pct:
            cat_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ffeeba")))
            cat_style.append(("TEXTCOLOR",  (4, i), (4,  i), DANGER))
            cat_style.append(("FONTNAME",   (4, i), (4,  i), "CustomFont-Bold"))
        elif i % 2 == 0:
            cat_style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))

    cat_table.setStyle(TableStyle(cat_style))
    story.append(cat_table)
    story.append(Spacer(1, 0.4*cm))


    story.append(Paragraph("AI-Powered Suggestions", styles["section"]))

    lines = suggestions.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.replace("**", "")
        story.append(Paragraph(line, styles["suggestion"]))

    story.append(Spacer(1, 0.4*cm))


    story.append(Paragraph("Transaction Details", styles["section"]))

    if not df.empty and "transaction_type" in df.columns:
        debits = df[df["transaction_type"] == "debit"].copy()
    else:
        debits = df.copy()

    txn_header = ["Date", "Description", "Amount", "Category", "Method"]
    txn_rows   = [txn_header]

    for _, row in debits.iterrows():
        txn_rows.append([
            str(row.get("date", "")),
            str(row.get("description", ""))[:28],
            f"₹ {row.get('amount', 0):,.2f}",
            str(row.get("category", "")),
            str(row.get("method", ""))
        ])

    txn_table = Table(
        txn_rows,
        colWidths=[W*0.18, W*0.32, W*0.18, W*0.18, W*0.14]
    )
    txn_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "CustomFont-Bold"),
        ("FONTSIZE",      (0, 0), (-1,-1), 8),
        ("ALIGN",         (2, 0), (2, -1), "RIGHT"),
        ("ALIGN",         (0, 0), (1, -1), "LEFT"),
        ("ALIGN",         (3, 0), (4, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
        ("GRID",          (0, 0), (-1,-1), 0.3, MID_GRAY),
        ("ROWHEIGHT",     (0, 0), (-1,-1), 18),
        ("TOPPADDING",    (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 4),
        ("FONTNAME",      (0, 1), (-1,-1), "CustomFont")
    ]
    for i in range(1, len(txn_rows)):
        if i % 2 == 0:
            txn_style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))

    txn_table.setStyle(TableStyle(txn_style))
    story.append(txn_table)


    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width=W, thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Generated by FinSight AI  |  Powered by DistilBERT + Groq LLaMA  |  For personal use only",
        styles["footer"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()