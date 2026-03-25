"""
PDF utilities optimized for low-memory environments (Render free tier).

Features:
- Extract text per page
- Render pages as PNG (memory safe)
- Extract selected pages into new PDF
- Stream thumbnails one-by-one for segregation
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple, Generator

import fitz  # PyMuPDF

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()


# -------------------------
# Basic helpers
# -------------------------

def get_page_count(pdf_bytes: bytes) -> int:
    """Return total number of pages."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


# -------------------------
# TEXT EXTRACTION
# -------------------------

def extract_text_per_page(pdf_bytes: bytes) -> Dict[int, str]:
    """
    Extract text page-by-page.
    Memory safe because pages are processed sequentially.
    Returns:
        {page_number (1-based): extracted_text}
    """

    texts: Dict[int, str] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

        for i in range(doc.page_count):

            page = doc.load_page(i)

            text = page.get_text("text")

            texts[i + 1] = text.strip()

            del page
            gc.collect()

    return texts


# -------------------------
# IMAGE RENDERING
# -------------------------

def render_page_as_png_bytes(
    page: fitz.Page,
    dpi: int
) -> bytes:
    """
    Render a single page to PNG bytes.
    Uses reduced DPI for memory safety.
    """

    # cap dpi to avoid memory spikes
    dpi = min(dpi, 120)

    mat = fitz.Matrix(dpi / 72, dpi / 72)

    pix = page.get_pixmap(
        matrix=mat,
        colorspace=fitz.csRGB
    )

    img_bytes = pix.tobytes("png")

    del pix

    return img_bytes


def render_pages_as_png(
    pdf_bytes: bytes,
    page_numbers: List[int],
    dpi: int | None = None,
) -> Dict[int, bytes]:
    """
    Render selected pages to PNG bytes.

    Memory safe:
    pages processed sequentially
    """

    dpi = dpi or settings.pdf_image_dpi

    result: Dict[int, bytes] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

        for page_num in page_numbers:

            if page_num < 1 or page_num > doc.page_count:
                logger.warning(
                    f"Page {page_num} out of range (total={doc.page_count})"
                )
                continue

            page = doc.load_page(page_num - 1)

            img_bytes = render_page_as_png_bytes(page, dpi)

            result[page_num] = img_bytes

            del page
            del img_bytes

            gc.collect()

    return result


# -------------------------
# STREAMING THUMBNAILS
# -------------------------

def stream_page_thumbnails(
    pdf_bytes: bytes,
    dpi: int | None = None,
) -> Generator[Tuple[int, bytes], None, None]:
    """
    Stream thumbnails one-by-one.

    Best for segregation agent.

    Example usage:

        for page_num, img in stream_page_thumbnails(pdf_bytes):
            classify(img)

    Uses very low memory.
    """

    dpi = dpi or settings.pdf_image_dpi

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

        total_pages = doc.page_count

        logger.info(f"[PDF] streaming {total_pages} thumbnails")

        for i in range(total_pages):

            page_number = i + 1

            page = doc.load_page(i)

            img_bytes = render_page_as_png_bytes(page, dpi)

            yield page_number, img_bytes

            del page
            del img_bytes

            gc.collect()


# -------------------------
# CREATE NEW PDF FROM SELECTED PAGES
# -------------------------

def extract_pages_as_bytes(
    pdf_bytes: bytes,
    page_numbers: List[int]
) -> bytes:
    """
    Create new PDF with only selected pages.

    Memory safe.
    """

    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:

        dst = fitz.open()

        for pn in sorted(page_numbers):

            if 1 <= pn <= src.page_count:

                dst.insert_pdf(
                    src,
                    from_page=pn - 1,
                    to_page=pn - 1
                )

        result = dst.tobytes()

        dst.close()

        return result


# -------------------------
# OPTIONAL HELPER
# -------------------------

def get_page_thumbnails_low_memory(
    pdf_bytes: bytes
) -> List[Tuple[int, bytes]]:
    """
    Compatibility wrapper if existing code expects list.

    WARNING:
    uses more memory than streaming version.
    Prefer stream_page_thumbnails().
    """

    thumbnails = []

    for page_num, img in stream_page_thumbnails(pdf_bytes):

        thumbnails.append((page_num, img))

    return thumbnails