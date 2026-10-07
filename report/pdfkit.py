"""Мини-набор для вёрстки PDF (reportlab): шрифты, стили, таблицы, рисунки."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "assets" / "fonts"

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#8a8984")
RULE = colors.HexColor("#e6e5e1")
TINT = colors.HexColor("#f4f3f0")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_T = colors.HexColor("#e8f1fc")


def register_fonts():
    for w in ["Regular", "Medium", "SemiBold", "Bold", "Italic"]:
        pdfmetrics.registerFont(TTFont(f"Inter-{w}", str(FONTS / f"Inter-{w}.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Inter", normal="Inter-Regular", bold="Inter-SemiBold",
                       italic="Inter-Italic", boldItalic="Inter-SemiBold")


def styles(base=9.6):
    s = {}
    s["body"] = ParagraphStyle("body", fontName="Inter-Regular", fontSize=base, leading=base * 1.45,
                               textColor=INK, alignment=TA_LEFT, spaceAfter=base * 0.55)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=base - 1.6, leading=(base - 1.6) * 1.4,
                                textColor=INK2, spaceAfter=3)
    s["cap"] = ParagraphStyle("cap", parent=s["small"], fontSize=base - 1.8, textColor=INK2, spaceBefore=2,
                              spaceAfter=10)
    s["h1"] = ParagraphStyle("h1", fontName="Inter-SemiBold", fontSize=base + 6.5, leading=(base + 6.5) * 1.2,
                             textColor=INK, spaceBefore=6, spaceAfter=8)
    s["h2"] = ParagraphStyle("h2", fontName="Inter-SemiBold", fontSize=base + 2.4, leading=(base + 2.4) * 1.3,
                             textColor=INK, spaceBefore=10, spaceAfter=5)
    s["h3"] = ParagraphStyle("h3", fontName="Inter-SemiBold", fontSize=base + 0.6, leading=(base + 0.6) * 1.35,
                             textColor=INK, spaceBefore=6, spaceAfter=3)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12, bulletIndent=2, spaceAfter=3)
    s["cell"] = ParagraphStyle("cell", fontName="Inter-Regular", fontSize=base - 1.4, leading=(base - 1.4) * 1.3,
                               textColor=INK)
    s["cellb"] = ParagraphStyle("cellb", parent=s["cell"], fontName="Inter-SemiBold")
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName="Inter-Medium", textColor=INK2)
    s["box"] = ParagraphStyle("box", parent=s["body"], spaceAfter=3)
    return s


def bullets(items, st):
    return [Paragraph(t, st["bullet"], bulletText="•") for t in items]


def table(rows, st, col_widths, header=True, bold_first_col=False, highlight_rows=(), align_right_from=1,
          zebra=False):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            style = st["cellh"] if (header and r == 0) else (
                st["cellb"] if (bold_first_col and c == 0) or r in highlight_rows else st["cell"])
            cells.append(Paragraph(str(v), style))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    ts = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2) if header else ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
          ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE)]
    for r in range(1 if header else 0, len(rows) - 1):
        ts.append(("LINEBELOW", (0, r), (-1, r), 0.3, RULE))
    for r in highlight_rows:
        ts.append(("BACKGROUND", (0, r), (-1, r), ACCENT_T))
    if zebra:
        for r in range(1, len(rows), 2):
            ts.append(("BACKGROUND", (0, r), (-1, r), TINT))
    t.setStyle(TableStyle(ts))
    for r in range(len(data)):
        for c in range(align_right_from, len(data[r])):
            data[r][c].style = ParagraphStyle(f"r{r}{c}", parent=data[r][c].style, alignment=2)
    return t


def figure(path, width, caption, st):
    from PIL import Image as PImage
    w, h = PImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, Paragraph(caption, st["cap"])])


def boxed(flowables, width, bg=TINT, pad=8):
    t = Table([[flowables]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), pad),
                           ("RIGHTPADDING", (0, 0), (-1, -1), pad), ("TOPPADDING", (0, 0), (-1, -1), pad),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
    return t


def fmt(x, d=0):
    """Число по-русски: пробел — разделитель тысяч, запятая — десятичный."""
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def fsign(x, d=0):
    """Число со знаком: +2,1 / −1,4."""
    return ("+" if x > 0 else "") + fmt(x, d)


def pct(x, d=0):
    return fmt(100 * x, d) + "%"


__all__ = ["register_fonts", "styles", "bullets", "table", "figure", "boxed", "fmt", "fsign", "pct", "Spacer", "Paragraph",
           "mm", "INK", "INK2", "MUTED", "RULE", "TINT", "ACCENT", "ACCENT_T"]
