"""
PDF report generator for MarketInsight AI.

Converts Markdown research reports into professionally formatted PDF
documents using FPDF2.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path


from fpdf import FPDF

logger = logging.getLogger(__name__)

# ── Markdown regex patterns ──────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+(.+)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HORIZONTAL_RULE_RE = re.compile(r"^---+\s*$")
_DATE_RE = re.compile(r"\*\*Generated:\*\*\s*(.+)$")
_TITLE_LINE_RE = re.compile(r"^#\s+(.+)$")

# ── Layout constants ─────────────────────────────────────────────
_MARGIN = 20
_BODY_SIZE = 11
_TITLE_SIZE = 24
_H1_SIZE = 18
_H2_SIZE = 14
_H3_SIZE = 12
_LEADING = 5


class _NumberedPDF(FPDF):
    """FPDF subclass that renders ``{nb}`` page numbers in the footer."""

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


class PDFGenerator:
    """Convert a Markdown market research report into a PDF document.

    Parameters
    ----------
    output_dir:
        Directory where generated PDFs are saved.  Created
        automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir

    # ── Public API ───────────────────────────────────────────────

    def generate(self, report_markdown: str, output_path: str) -> str:
        """Render *report_markdown* into a PDF and write to *output_path*.

        Parameters
        ----------
        report_markdown:
            Full Markdown report content.
        output_path:
            Destination path for the generated PDF file.

        Returns
        -------
        str
            Absolute path to the generated PDF.

        Raises
        ------
        RuntimeError
            If PDF generation fails unexpectedly.
        """
        resolved_path = Path(output_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating PDF: %s (%d chars)",
            resolved_path.name,
            len(report_markdown),
        )

        try:
            pdf = _build_pdf(report_markdown)
            pdf.output(str(resolved_path))
        except Exception as exc:
            logger.exception("PDF generation failed")
            raise RuntimeError(f"PDF generation failed: {exc}") from exc

        logger.info("PDF written to %s", resolved_path)
        return str(resolved_path)


# ────────────────────────────────────────────────────────────────
# Internal PDF builder
# ────────────────────────────────────────────────────────────────


def _build_pdf(markdown: str) -> _NumberedPDF:
    """Construct a fully rendered ``_NumberedPDF`` from *markdown*."""
    lines = markdown.splitlines()
    title, subtitle, generated_date = _extract_metadata(lines)

    pdf = _NumberedPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margin(_MARGIN)

    _add_title_page(pdf, title, subtitle, generated_date)
    pdf.add_page()
    _render_body(lines, pdf)

    return pdf


def _extract_metadata(
    lines: list[str],
) -> tuple[str, str, str]:
    """Extract title, subtitle, and generated date from the Markdown."""
    title = "Market Research Report"
    subtitle_parts: list[str] = []
    generated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for line in lines:
        stripped = line.strip()

        m = _TITLE_LINE_RE.match(stripped)
        if m:
            title = m.group(1).strip()
            continue

        if stripped.startswith("**Industry:**") or stripped.startswith("**Country"):
            subtitle_parts.append(stripped)
            continue

        m = _DATE_RE.search(stripped)
        if m:
            generated_date = m.group(1).strip()

    subtitle = " | ".join(subtitle_parts)
    return title, subtitle, generated_date


def _add_title_page(
    pdf: _NumberedPDF,
    title: str,
    subtitle: str,
    generated_date: str,
) -> None:
    """Render a professional title page."""
    pdf.add_page()

    pdf.ln(60)
    pdf.set_font("Helvetica", "B", _TITLE_SIZE)
    pdf.multi_cell(0, 12, title, align="C")
    pdf.ln(8)

    if subtitle:
        pdf.set_font("Helvetica", "", _BODY_SIZE)
        pdf.multi_cell(0, 8, subtitle, align="C")
        pdf.ln(16)

    pdf.set_font("Helvetica", "", _BODY_SIZE)
    pdf.cell(0, 10, f"Generated: {generated_date}", align="C")
    pdf.ln(20)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 10, "MarketInsight AI - Automated Market Research", align="C")


def _cw(pdf: _NumberedPDF) -> float:
    """Return the available content width between margins."""
    return pdf.w - pdf.l_margin - pdf.r_margin


def _render_body(lines: list[str], pdf: _NumberedPDF) -> None:
    """Render the main Markdown content into the PDF."""
    heading_count = 0

    for line in lines:
        stripped = line.strip()

        # Skip the first ``# Title`` (already on title page).
        if heading_count == 0 and _TITLE_LINE_RE.match(stripped):
            heading_count += 1
            continue

        if _HORIZONTAL_RULE_RE.match(stripped):
            continue

        if not stripped:
            pdf.ln(_LEADING)
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            heading_count += 1
            level = len(heading.group(1))
            text = _strip_inline(heading.group(2))
            _render_heading(pdf, text, level)
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            text = _strip_inline(bullet.group(1))
            if not text:
                continue
            pdf.set_font("Helvetica", "", _BODY_SIZE)
            pdf.cell(10)
            pdf.multi_cell(_cw(pdf) - 10, 6, f"\u002d {text}")
            continue

        # Regular paragraph.
        text = _strip_inline(stripped)
        if not text:
            continue
        pdf.set_font("Helvetica", "", _BODY_SIZE)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(_cw(pdf), 6, text)


def _render_heading(pdf: _NumberedPDF, text: str, level: int) -> None:
    """Add a heading at the given hierarchy level."""
    if not text:
        return
    sizes = {1: _H1_SIZE, 2: _H2_SIZE, 3: _H3_SIZE}
    size = sizes.get(level, _H2_SIZE)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(_cw(pdf), size * 0.6, text)
    pdf.ln(2)


def _strip_inline(text: str) -> str:
    """Remove inline Markdown formatting (bold, italic)."""
    text = _BOLD_RE.sub(r"\1", text)
    return text.strip()
