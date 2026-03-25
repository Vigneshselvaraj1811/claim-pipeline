"""
Tests for the Claim Processing Pipeline.
Run with: pytest tests/ -v
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.models import (
    BillItem,
    DischargeSummaryData,
    IdentityData,
    ItemizedBillData,
    PageClassification,
    PipelineState,
    SegregatorOutput,
)
from app.utils.json_parser import extract_json


# ---------------------------------------------------------------------------
# JSON parser tests
# ---------------------------------------------------------------------------

class TestJsonParser:
    def test_parse_direct_json(self):
        raw = '{"key": "value", "num": 42}'
        result = extract_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_parse_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_parse_embedded_json(self):
        raw = 'Some text before {"key": "value"} some text after'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all!!!")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPipelineState:
    def test_default_state(self):
        state = PipelineState(claim_id="TEST-001", pdf_bytes=b"")
        assert state.claim_id == "TEST-001"
        assert state.errors == []
        assert state.final_result is None

    def test_state_with_errors(self):
        state = PipelineState(claim_id="TEST-002", pdf_bytes=b"")
        state.errors.append("test error")
        assert len(state.errors) == 1


class TestBillItem:
    def test_bill_item_creation(self):
        item = BillItem(description="Room Charges", amount=1000.0)
        assert item.amount == 1000.0
        assert item.description == "Room Charges"


# ---------------------------------------------------------------------------
# Segregator tests
# ---------------------------------------------------------------------------

class TestSegregator:
    def _make_state(self) -> PipelineState:
        import fitz
        # Create a minimal in-memory PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "MEDICAL CLAIM FORM\nPatient: John Smith")
        pdf_bytes = doc.tobytes()
        doc.close()
        return PipelineState(claim_id="TEST-001", pdf_bytes=pdf_bytes)

    @patch("app.agents.segregator.get_llm_client")
    def test_segregator_classifies_pages(self, mock_llm_factory):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps([
            {"page_number": 1, "doc_type": "claim_forms", "confidence": 0.95, "reasoning": "test"}
        ])
        mock_llm_factory.return_value = mock_llm

        from app.agents.segregator import segregator_node
        state = self._make_state()
        result = segregator_node(state)

        assert result.segregator_output is not None
        assert "claim_forms" in result.segregator_output.page_map
        assert result.segregator_output.page_map["claim_forms"] == [1]

    @patch("app.agents.segregator.get_llm_client")
    def test_segregator_fallback_on_error(self, mock_llm_factory):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("LLM timeout")
        mock_llm_factory.return_value = mock_llm

        from app.agents.segregator import segregator_node
        state = self._make_state()
        result = segregator_node(state)

        assert result.segregator_output is not None
        assert len(result.errors) > 0
        # Fallback: all pages → "other"
        assert "other" in result.segregator_output.page_map


# ---------------------------------------------------------------------------
# ID Agent tests
# ---------------------------------------------------------------------------

class TestIDAgent:
    def _make_state_with_segregation(self) -> PipelineState:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "GOVERNMENT ID CARD\nName: John Smith\nDOB: 1985-03-15")
        pdf_bytes = doc.tobytes()
        doc.close()

        state = PipelineState(claim_id="TEST-002", pdf_bytes=pdf_bytes)
        state.segregator_output = SegregatorOutput(
            classifications=[
                PageClassification(page_number=1, doc_type="identity_document", confidence=0.95)
            ],
            page_map={"identity_document": [1]},
        )
        return state

    @patch("app.agents.id_agent.get_llm_client")
    def test_id_agent_extracts_data(self, mock_llm_factory):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "patient_name": "John Michael Smith",
            "date_of_birth": "March 15, 1985",
            "id_numbers": {"government_id": "ID-987-654-321"},
            "policy_details": {},
            "address": None,
            "contact": None,
        })
        mock_llm_factory.return_value = mock_llm

        from app.agents.id_agent import id_agent_node
        state = self._make_state_with_segregation()
        result = id_agent_node(state)

        assert result.identity_data is not None
        assert result.identity_data.patient_name == "John Michael Smith"
        assert result.identity_data.raw_pages == [1]

    def test_id_agent_no_pages(self):
        import fitz
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        state = PipelineState(claim_id="TEST-003", pdf_bytes=pdf_bytes)
        state.segregator_output = SegregatorOutput(
            classifications=[], page_map={}
        )

        from app.agents.id_agent import id_agent_node
        result = id_agent_node(state)

        assert result.identity_data is not None
        assert result.identity_data.patient_name is None


# ---------------------------------------------------------------------------
# Itemized Bill Agent tests
# ---------------------------------------------------------------------------

class TestItemizedBillAgent:
    @patch("app.agents.itemized_bill_agent.get_llm_client")
    def test_bill_total_calculated(self, mock_llm_factory):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "bill_number": "BILL-2025-789456",
            "bill_date": "February 1, 2025",
            "patient_name": "John Michael Smith",
            "items": [
                {"description": "Room Charges", "quantity": 5, "unit_rate": 200.0, "amount": 1000.0},
                {"description": "Consultation", "quantity": 1, "unit_rate": 150.0, "amount": 150.0},
            ],
            "subtotal": 1150.0,
            "tax": 57.5,
            "total_amount": 1207.5,
            "insurance_payment": 966.0,
            "patient_responsibility": 241.5,
        })
        mock_llm_factory.return_value = mock_llm

        import fitz
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        state = PipelineState(claim_id="TEST-004", pdf_bytes=pdf_bytes)
        state.segregator_output = SegregatorOutput(
            classifications=[
                PageClassification(page_number=1, doc_type="itemized_bill", confidence=0.9)
            ],
            page_map={"itemized_bill": [1]},
        )

        from app.agents.itemized_bill_agent import itemized_bill_agent_node
        result = itemized_bill_agent_node(state)

        assert result.itemized_bill_data is not None
        assert result.itemized_bill_data.total_amount == 1207.5
        assert len(result.itemized_bill_data.items) == 2


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------

class TestAggregator:
    def test_aggregator_builds_result(self):
        state = PipelineState(claim_id="TEST-005", pdf_bytes=b"")
        state.segregator_output = SegregatorOutput(
            classifications=[
                PageClassification(page_number=1, doc_type="identity_document", confidence=0.9),
            ],
            page_map={"identity_document": [1]},
        )
        state.identity_data = IdentityData(patient_name="John Smith", raw_pages=[1])
        state.discharge_summary_data = DischargeSummaryData(
            admission_diagnosis="Pneumonia", raw_pages=[]
        )
        state.itemized_bill_data = ItemizedBillData(total_amount=1000.0, raw_pages=[])

        from app.agents.aggregator import aggregator_node
        result = aggregator_node(state)

        assert result.final_result is not None
        assert result.final_result["claim_id"] == "TEST-005"
        assert result.final_result["identity"]["patient_name"] == "John Smith"
        assert result.final_result["discharge_summary"]["admission_diagnosis"] == "Pneumonia"
        assert result.final_result["itemized_bill"]["total_amount"] == 1000.0


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------

class TestAPIRoute:
    @patch("app.api.routes.run_pipeline")
    def test_process_endpoint_success(self, mock_pipeline):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_pipeline.return_value = {
            "claim_id": "CLM-001",
            "processing_summary": {"errors": [], "total_pages": 1, "document_types_found": [], "agents_executed": []},
            "identity": {"patient_name": "John Smith"},
            "discharge_summary": {},
            "itemized_bill": {},
            "document_classification": [],
        }

        client = TestClient(app)

        import io
        import fitz
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        response = client.post(
            "/api/process",
            data={"claim_id": "CLM-001"},
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["claim_id"] == "CLM-001"
        assert body["status"] == "success"

    @patch("app.api.routes.run_pipeline")
    def test_process_endpoint_rejects_non_pdf(self, mock_pipeline):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        import io
        response = client.post(
            "/api/process",
            data={"claim_id": "CLM-002"},
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
        )

        assert response.status_code == 400
