"""
Discharge Summary Agent
=======================
Extracts clinical information from pages classified as:
  - discharge_summary
"""
from __future__ import annotations

from app.models import DischargeSummaryData, PipelineState
from app.utils.llm_client import get_llm_client
from app.utils.pdf_utils import render_pages_as_png, extract_text_per_page
from app.utils.json_parser import extract_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical discharge summary extraction specialist.
Extract all clinical and administrative information from hospital discharge summaries.
Return ONLY valid JSON. No markdown. No extra text.
"""

USER_PROMPT = """Extract discharge summary information from these medical document pages.

Pages processed: {page_numbers}
Extracted text:
---
{page_text}
---

Return this exact JSON structure (use null for missing fields):
{{
  "admission_diagnosis": "...",
  "final_diagnosis": "...",
  "admission_date": "...",
  "discharge_date": "...",
  "length_of_stay": "...",
  "attending_physician": "...",
  "hospital_name": "...",
  "hospital_address": "...",
  "patient_name": "...",
  "mrn": "...",
  "date_of_birth": "...",
  "discharge_condition": "...",
  "discharge_medications": ["medication 1", "medication 2"],
  "follow_up_instructions": "...",
  "clinical_summary": "...",
  "procedures_performed": ["procedure 1"],
  "icd_codes": ["code1"]
}}
"""


def discharge_summary_agent_node(state: PipelineState) -> PipelineState:
    """LangGraph node: extract discharge summary data."""
    logger.info(f"[DischargeSummaryAgent] Starting for claim_id={state.claim_id}")

    if state.segregator_output is None:
        state.errors.append("DischargeSummaryAgent: segregator output missing")
        state.discharge_summary_data = DischargeSummaryData()
        return state

    page_map = state.segregator_output.page_map
    relevant_pages = sorted(set(page_map.get("discharge_summary", [])))

    if not relevant_pages:
        logger.info("[DischargeSummaryAgent] No discharge summary pages found")
        state.discharge_summary_data = DischargeSummaryData()
        return state

    logger.info(f"[DischargeSummaryAgent] Processing pages: {relevant_pages}")

    try:
        llm = get_llm_client()

        all_texts = extract_text_per_page(state.pdf_bytes)
        page_text = "\n\n".join(
            f"--- Page {p} ---\n{all_texts.get(p, '')}" for p in relevant_pages
        )

        images_map = render_pages_as_png(state.pdf_bytes, relevant_pages)
        images = [images_map[p] for p in relevant_pages if p in images_map]

        user_prompt = USER_PROMPT.format(
            page_numbers=relevant_pages,
            page_text=page_text[:3000],
        )

        raw = llm.complete(SYSTEM_PROMPT, user_prompt, images=images if images else None)
        data = extract_json(raw)

        state.discharge_summary_data = DischargeSummaryData(
            admission_diagnosis=data.get("admission_diagnosis") or data.get("final_diagnosis"),
            admission_date=data.get("admission_date"),
            discharge_date=data.get("discharge_date"),
            length_of_stay=data.get("length_of_stay"),
            attending_physician=data.get("attending_physician"),
            hospital_name=data.get("hospital_name"),
            discharge_condition=data.get("discharge_condition"),
            discharge_medications=data.get("discharge_medications") or [],
            follow_up_instructions=data.get("follow_up_instructions"),
            clinical_summary=data.get("clinical_summary"),
            raw_pages=relevant_pages,
        )
        logger.info(
            f"[DischargeSummaryAgent] Diagnosis: {state.discharge_summary_data.admission_diagnosis}"
        )

    except Exception as e:
        logger.error(f"[DischargeSummaryAgent] Extraction failed: {e}")
        state.errors.append(f"DischargeSummaryAgent error: {e}")
        state.discharge_summary_data = DischargeSummaryData(raw_pages=relevant_pages)

    return state
