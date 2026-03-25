"""
Segregator Agent - processes one page at a time to avoid Groq 413 limits.
"""
from __future__ import annotations

import json
from typing import Dict, List

from app.models import PageClassification, PipelineState, SegregatorOutput, DOCUMENT_TYPES
from app.utils.llm_client import get_llm_client
from app.utils.pdf_utils import render_pages_as_png, extract_text_per_page
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical document classifier.
Classify the given page into exactly one of these document types:
- claim_forms
- cheque_or_bank_details
- identity_document
- itemized_bill
- discharge_summary
- prescription
- investigation_report
- cash_receipt
- other

Respond ONLY with valid JSON. No markdown. No extra text.
"""

USER_PROMPT = """Classify this single page from a medical claim PDF.

Page number: {page_number} of {total_pages}

Respond with exactly this JSON:
{{
  "page_number": {page_number},
  "doc_type": "<one of the 9 types>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}}
"""


def segregator_node(state: PipelineState) -> PipelineState:
    logger.info(f"[Segregator] Starting for claim_id={state.claim_id}")

    llm = get_llm_client()
    page_texts = extract_text_per_page(state.pdf_bytes)
    total_pages = len(page_texts)
    logger.info(f"[Segregator] Total pages: {total_pages}")

    # Render all pages as images (one at a time usage)
    all_page_nums = list(range(1, total_pages + 1))
    images_map = render_pages_as_png(state.pdf_bytes, all_page_nums)

    classifications: List[PageClassification] = []

    for page_num in all_page_nums:
        try:
            cls = _classify_single_page(
                llm=llm,
                page_num=page_num,
                total_pages=total_pages,
                page_text=page_texts.get(page_num, ""),
                image=images_map.get(page_num),
            )
            classifications.append(cls)
            logger.info(f"[Segregator] Page {page_num}/{total_pages} → {cls.doc_type} ({cls.confidence:.2f})")
        except Exception as e:
            logger.error(f"[Segregator] Page {page_num} failed: {e}")
            state.errors.append(f"Segregator page {page_num} error: {e}")
            classifications.append(
                PageClassification(page_number=page_num, doc_type="other", confidence=0.0, reasoning="error")
            )

    # Build page_map
    page_map: Dict[str, List[int]] = {}
    for cls in classifications:
        dt = cls.doc_type if cls.doc_type in DOCUMENT_TYPES else "other"
        page_map.setdefault(dt, []).append(cls.page_number)

    state.segregator_output = SegregatorOutput(
        classifications=classifications,
        page_map=page_map,
    )

    logger.info(f"[Segregator] Done: { {k: len(v) for k, v in page_map.items()} }")
    return state


def _classify_single_page(
    llm,
    page_num: int,
    total_pages: int,
    page_text: str,
    image: bytes | None,
) -> PageClassification:

    user_prompt = USER_PROMPT.format(
        page_number=page_num,
        total_pages=total_pages,
    )

    # Add text hint if available
    if page_text.strip():
        user_prompt += f"\n\nExtracted text from page:\n---\n{page_text[:500]}\n---"

    # Send single image only
    images = [image] if image else None
    raw = llm.complete(SYSTEM_PROMPT, user_prompt, images=images)

    return _parse_single(raw, page_num)


def _parse_single(raw: str, page_num: int) -> PageClassification:
    import re

    # Try direct parse
    try:
        data = json.loads(raw)
        return _to_cls(data, page_num)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            return _to_cls(data, page_num)
        except json.JSONDecodeError:
            pass

    # Find first {...}
    match = re.search(r"\{[\s\S]*?\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return _to_cls(data, page_num)
        except json.JSONDecodeError:
            pass

    logger.warning(f"[Segregator] Could not parse response for page {page_num}: {raw[:200]}")
    return PageClassification(page_number=page_num, doc_type="other", confidence=0.0, reasoning="parse_error")


def _to_cls(data: dict, page_num: int) -> PageClassification:
    doc_type = data.get("doc_type", "other")
    if doc_type not in DOCUMENT_TYPES:
        doc_type = "other"
    return PageClassification(
        page_number=int(data.get("page_number", page_num)),
        doc_type=doc_type,
        confidence=float(data.get("confidence", 0.5)),
        reasoning=str(data.get("reasoning", "")),
    )