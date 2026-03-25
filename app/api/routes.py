"""
FastAPI routes for the Claim Processing Pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.graph.pipeline import run_pipeline
from app.models import ProcessResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()

MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post(
    "/process",
    response_model=ProcessResponse,
    summary="Process a medical claim PDF",
    description=(
        "Upload a PDF claim file with a claim_id. "
        "The pipeline classifies pages, extracts data via multiple agents, "
        "and returns structured JSON."
    ),
)
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF file to process"),
) -> ProcessResponse:
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.max_upload_size_mb} MB limit.",
        )

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    logger.info(
        f"[Route] Processing claim_id={claim_id}, "
        f"file={file.filename}, size={len(pdf_bytes)} bytes"
    )

    start = time.perf_counter()

    try:
        result: Dict[str, Any] = run_pipeline(claim_id=claim_id, pdf_bytes=pdf_bytes)
    except Exception as e:
        logger.exception(f"[Route] Unhandled pipeline error for claim_id={claim_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}",
        )

    elapsed = time.perf_counter() - start
    logger.info(f"[Route] Completed claim_id={claim_id} in {elapsed:.2f}s")

    errors = result.get("processing_summary", {}).get("errors", [])
    pipeline_status = "success" if not errors else "partial"

    return ProcessResponse(
        claim_id=claim_id,
        status=pipeline_status,
        data=result,
        errors=errors,
        page_classification=result.get("document_classification"),
    )
