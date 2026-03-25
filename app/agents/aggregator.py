"""
Aggregator Node
===============
Combines outputs from all extraction agents into a unified final JSON.
"""
from __future__ import annotations

from typing import Any, Dict

from app.models import PipelineState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def aggregator_node(state: PipelineState) -> PipelineState:
    """LangGraph node: aggregate all agent results."""
    logger.info(f"[Aggregator] Building final result for claim_id={state.claim_id}")

    result: Dict[str, Any] = {
        "claim_id": state.claim_id,
        "processing_summary": _build_summary(state),
        "identity": _serialize_identity(state),
        "discharge_summary": _serialize_discharge(state),
        "itemized_bill": _serialize_bill(state),
        "document_classification": _serialize_classification(state),
    }

    state.final_result = result
    logger.info("[Aggregator] Final result assembled successfully")
    return state


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _build_summary(state: PipelineState) -> Dict[str, Any]:
    page_map = state.segregator_output.page_map if state.segregator_output else {}
    total_pages = (
        len(state.segregator_output.classifications)
        if state.segregator_output
        else 0
    )

    agents_ran = []
    if state.identity_data:
        agents_ran.append("id_agent")
    if state.discharge_summary_data:
        agents_ran.append("discharge_summary_agent")
    if state.itemized_bill_data:
        agents_ran.append("itemized_bill_agent")

    return {
        "total_pages": total_pages,
        "document_types_found": list(page_map.keys()),
        "agents_executed": agents_ran,
        "errors": state.errors,
    }


def _serialize_identity(state: PipelineState) -> Dict[str, Any]:
    if not state.identity_data:
        return {}
    d = state.identity_data
    return {
        "patient_name": d.patient_name,
        "date_of_birth": d.date_of_birth,
        "id_numbers": d.id_numbers,
        "policy_details": d.policy_details,
        "address": d.address,
        "contact": d.contact,
        "source_pages": d.raw_pages,
    }


def _serialize_discharge(state: PipelineState) -> Dict[str, Any]:
    if not state.discharge_summary_data:
        return {}
    d = state.discharge_summary_data
    return {
        "hospital_name": d.hospital_name,
        "admission_diagnosis": d.admission_diagnosis,
        "admission_date": d.admission_date,
        "discharge_date": d.discharge_date,
        "length_of_stay": d.length_of_stay,
        "attending_physician": d.attending_physician,
        "discharge_condition": d.discharge_condition,
        "discharge_medications": d.discharge_medications,
        "follow_up_instructions": d.follow_up_instructions,
        "clinical_summary": d.clinical_summary,
        "source_pages": d.raw_pages,
    }


def _serialize_bill(state: PipelineState) -> Dict[str, Any]:
    if not state.itemized_bill_data:
        return {}
    d = state.itemized_bill_data
    return {
        "bill_number": d.bill_number,
        "bill_date": d.bill_date,
        "patient_name": d.patient_name,
        "items": [
            {
                "date": item.date,
                "description": item.description,
                "quantity": item.quantity,
                "unit_rate": item.unit_rate,
                "amount": item.amount,
            }
            for item in d.items
        ],
        "subtotal": d.subtotal,
        "tax": d.tax,
        "total_amount": d.total_amount,
        "insurance_payment": d.insurance_payment,
        "patient_responsibility": d.patient_responsibility,
        "source_pages": d.raw_pages,
    }


def _serialize_classification(state: PipelineState) -> list:
    if not state.segregator_output:
        return []
    return [
        {
            "page_number": c.page_number,
            "doc_type": c.doc_type,
            "confidence": c.confidence,
            "reasoning": c.reasoning,
        }
        for c in state.segregator_output.classifications
    ]
