"""
ID Agent
========
Extracts identity information from pages classified as:
  - identity_document
  - claim_forms  (patient info section)
"""
from __future__ import annotations

from app.models import IdentityData, PipelineState
from app.utils.llm_client import get_llm_client
from app.utils.pdf_utils import render_pages_as_png, extract_text_per_page
from app.utils.json_parser import extract_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical identity extraction specialist.
Extract all identity and patient information from the provided document pages.
Return ONLY valid JSON. No markdown. No extra text.
"""

USER_PROMPT = """Extract identity information from these medical document pages.

Pages processed: {page_numbers}
Extracted text:
---
{page_text}
---

Return this exact JSON structure (use null for missing fields):
{{
  "patient_name": "...",
  "date_of_birth": "...",
  "id_numbers": {{
    "government_id": "...",
    "patient_id": "...",
    "policy_number": "...",
    "mrn": "...",
    "any_other_id_label": "value"
  }},
  "policy_details": {{
    "insurance_provider": "...",
    "policy_number": "...",
    "group_number": "...",
    "subscriber_name": "...",
    "effective_date": "..."
  }},
  "address": "...",
  "contact": "...",
  "email": "...",
  "gender": "...",
  "blood_group": "..."
}}
"""


def id_agent_node(state: PipelineState) -> PipelineState:
    """LangGraph node: extract identity data."""
    logger.info(f"[IDAgent] Starting for claim_id={state.claim_id}")

    if state.segregator_output is None:
        state.errors.append("IDAgent: segregator output missing")
        state.identity_data = IdentityData()
        return state

    page_map = state.segregator_output.page_map

    # Collect relevant pages
    target_types = ["identity_document", "claim_forms"]
    relevant_pages: list[int] = []
    for dt in target_types:
        relevant_pages.extend(page_map.get(dt, []))

    relevant_pages = sorted(set(relevant_pages))

    if not relevant_pages:
        logger.info("[IDAgent] No identity/claim pages found")
        state.identity_data = IdentityData()
        return state

    logger.info(f"[IDAgent] Processing pages: {relevant_pages}")

    try:
        llm = get_llm_client()

        # Get text for these pages
        all_texts = extract_text_per_page(state.pdf_bytes)
        page_text = "\n\n".join(
            f"--- Page {p} ---\n{all_texts.get(p, '')}" for p in relevant_pages
        )

        # Render pages as images for vision
        images_map = render_pages_as_png(state.pdf_bytes, relevant_pages)
        images = [images_map[p] for p in relevant_pages if p in images_map]

        user_prompt = USER_PROMPT.format(
            page_numbers=relevant_pages,
            page_text=page_text[:3000],
        )

        raw = llm.complete(SYSTEM_PROMPT, user_prompt, images=images if images else None)
        data = extract_json(raw)

        state.identity_data = IdentityData(
            patient_name=data.get("patient_name"),
            date_of_birth=data.get("date_of_birth"),
            id_numbers=data.get("id_numbers") or {},
            policy_details=data.get("policy_details") or {},
            address=data.get("address"),
            contact=data.get("contact"),
            raw_pages=relevant_pages,
        )
        logger.info(f"[IDAgent] Extracted patient: {state.identity_data.patient_name}")

    except Exception as e:
        logger.error(f"[IDAgent] Extraction failed: {e}")
        state.errors.append(f"IDAgent error: {e}")
        state.identity_data = IdentityData(raw_pages=relevant_pages)

    return state
