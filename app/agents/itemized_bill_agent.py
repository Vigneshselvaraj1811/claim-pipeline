"""
Itemized Bill Agent
===================
Extracts all itemized charges from pages classified as:
  - itemized_bill
  - cash_receipt
"""
from __future__ import annotations

from app.models import BillItem, ItemizedBillData, PipelineState
from app.utils.llm_client import get_llm_client
from app.utils.pdf_utils import render_pages_as_png, extract_text_per_page
from app.utils.json_parser import extract_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical billing extraction specialist.
Extract ALL line items and totals from medical bills, hospital invoices, and receipts.
Be precise with amounts. Return ONLY valid JSON. No markdown. No extra text.
"""

USER_PROMPT = """Extract all itemized billing information from these medical document pages.

Pages processed: {page_numbers}
Extracted text:
---
{page_text}
---

Return this exact JSON structure (use null for missing fields, 0 for missing amounts):
{{
  "bill_number": "...",
  "bill_date": "...",
  "patient_name": "...",
  "patient_id": "...",
  "hospital_name": "...",
  "admission_date": "...",
  "discharge_date": "...",
  "insurance_provider": "...",
  "items": [
    {{
      "date": "...",
      "description": "Room Charges - Semi-Private",
      "quantity": 5,
      "unit_rate": 200.00,
      "amount": 1000.00
    }}
  ],
  "subtotal": 6113.00,
  "tax": 305.65,
  "tax_rate": "5%",
  "total_amount": 6418.65,
  "insurance_payment": 5134.92,
  "patient_responsibility": 1283.73,
  "payment_method": "..."
}}

IMPORTANT: Extract EVERY line item. Do not summarize or omit items.
"""


def itemized_bill_agent_node(state: PipelineState) -> PipelineState:
    """LangGraph node: extract itemized bill data."""
    logger.info(f"[ItemizedBillAgent] Starting for claim_id={state.claim_id}")

    if state.segregator_output is None:
        state.errors.append("ItemizedBillAgent: segregator output missing")
        state.itemized_bill_data = ItemizedBillData()
        return state

    page_map = state.segregator_output.page_map

    # Collect bill + receipt pages
    target_types = ["itemized_bill", "cash_receipt"]
    relevant_pages: list[int] = []
    for dt in target_types:
        relevant_pages.extend(page_map.get(dt, []))

    relevant_pages = sorted(set(relevant_pages))

    if not relevant_pages:
        logger.info("[ItemizedBillAgent] No bill pages found")
        state.itemized_bill_data = ItemizedBillData()
        return state

    logger.info(f"[ItemizedBillAgent] Processing pages: {relevant_pages}")

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
            page_text=page_text[:4000],
        )

        raw = llm.complete(SYSTEM_PROMPT, user_prompt, images=images if images else None)
        data = extract_json(raw)

        # Parse items
        raw_items = data.get("items") or []
        bill_items = []
        for item in raw_items:
            try:
                bill_items.append(
                    BillItem(
                        date=item.get("date"),
                        description=str(item.get("description", "Unknown")),
                        quantity=_to_float(item.get("quantity")),
                        unit_rate=_to_float(item.get("unit_rate")),
                        amount=_to_float(item.get("amount")) or 0.0,
                    )
                )
            except Exception:
                continue

        total_amount = _to_float(data.get("total_amount")) or sum(i.amount for i in bill_items)

        state.itemized_bill_data = ItemizedBillData(
            bill_number=data.get("bill_number"),
            bill_date=data.get("bill_date"),
            patient_name=data.get("patient_name"),
            items=bill_items,
            subtotal=_to_float(data.get("subtotal")),
            tax=_to_float(data.get("tax")),
            total_amount=total_amount,
            insurance_payment=_to_float(data.get("insurance_payment")),
            patient_responsibility=_to_float(data.get("patient_responsibility")),
            raw_pages=relevant_pages,
        )
        logger.info(
            f"[ItemizedBillAgent] {len(bill_items)} items, total={total_amount}"
        )

    except Exception as e:
        logger.error(f"[ItemizedBillAgent] Extraction failed: {e}")
        state.errors.append(f"ItemizedBillAgent error: {e}")
        state.itemized_bill_data = ItemizedBillData(raw_pages=relevant_pages)

    return state


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None
