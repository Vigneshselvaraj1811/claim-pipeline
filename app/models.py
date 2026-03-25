"""
Pydantic models for pipeline state, agent outputs, and API contracts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]


class PageClassification(BaseModel):
    page_number: int
    doc_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class SegregatorOutput(BaseModel):
    classifications: List[PageClassification]
    page_map: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Maps doc_type -> list of 1-based page numbers",
    )


# ---------------------------------------------------------------------------
# Extraction agent outputs
# ---------------------------------------------------------------------------

class IdentityData(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_numbers: Dict[str, Optional[str]] = Field(default_factory=dict)
    policy_details: Dict[str, Optional[str]] = Field(default_factory=dict)
    address: Optional[str] = None
    contact: Optional[str] = None
    raw_pages: List[int] = Field(default_factory=list)

class DischargeSummaryData(BaseModel):
    admission_diagnosis: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    length_of_stay: Optional[str] = None
    attending_physician: Optional[str] = None
    hospital_name: Optional[str] = None
    discharge_condition: Optional[str] = None
    discharge_medications: List[str] = Field(default_factory=list)
    follow_up_instructions: Optional[str] = None
    clinical_summary: Optional[str] = None
    raw_pages: List[int] = Field(default_factory=list)


class BillItem(BaseModel):
    date: Optional[str] = None
    description: str
    quantity: Optional[float] = None
    unit_rate: Optional[float] = None
    amount: float


class ItemizedBillData(BaseModel):
    bill_number: Optional[str] = None
    bill_date: Optional[str] = None
    patient_name: Optional[str] = None
    items: List[BillItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total_amount: float = 0.0
    insurance_payment: Optional[float] = None
    patient_responsibility: Optional[float] = None
    raw_pages: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph pipeline state
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Shared mutable state flowing through the LangGraph nodes."""

    claim_id: str
    pdf_bytes: bytes = b""

    # Filled by Segregator
    segregator_output: Optional[SegregatorOutput] = None

    # Filled by extraction agents
    identity_data: Optional[IdentityData] = None
    discharge_summary_data: Optional[DischargeSummaryData] = None
    itemized_bill_data: Optional[ItemizedBillData] = None

    # Final aggregated result
    final_result: Optional[Dict[str, Any]] = None

    # Error tracking
    errors: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ProcessResponse(BaseModel):
    claim_id: str
    status: str                          # "success" | "partial" | "error"
    data: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)
    page_classification: Optional[List[Dict[str, Any]]] = None
