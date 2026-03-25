"""
PDF utilities: extract text per page and render pages as PNG images.
Uses PyMuPDF (fitz) for both text and image extraction.
"""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()


def get_page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def extract_text_per_page(pdf_bytes: bytes) -> Dict[int, str]:
    """Return {1-based page number: text}."""
    texts: Dict[int, str] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            texts[i] = page.get_text("text").strip()
    return texts


def render_pages_as_png(
    pdf_bytes: bytes,
    page_numbers: List[int],
    dpi: int | None = None,
) -> Dict[int, bytes]:
    """
    Render specific pages (1-based) to PNG bytes.
    Returns {page_number: png_bytes}.
    """
    dpi = dpi or settings.pdf_image_dpi
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    result: Dict[int, bytes] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in page_numbers:
            if page_num < 1 or page_num > doc.page_count:
                logger.warning(f"Page {page_num} out of range (total={doc.page_count})")
                continue
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            result[page_num] = pix.tobytes("png")

    return result


def extract_pages_as_bytes(pdf_bytes: bytes, page_numbers: List[int]) -> bytes:
    """
    Create a NEW pdf containing only the specified pages (1-based).
    Returns the new PDF bytes.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        dst = fitz.open()
        for pn in sorted(page_numbers):
            if 1 <= pn <= src.page_count:
                dst.insert_pdf(src, from_page=pn - 1, to_page=pn - 1)
        return dst.tobytes()


def get_page_thumbnails(pdf_bytes: bytes) -> List[Tuple[int, bytes]]:
    """Render ALL pages as small thumbnails for segregation."""
    page_count = get_page_count(pdf_bytes)
    rendered = render_pages_as_png(
        pdf_bytes,
        list(range(1, page_count + 1)),
        dpi=settings.pdf_image_dpi,
    )
    return [(pn, rendered[pn]) for pn in sorted(rendered)]
