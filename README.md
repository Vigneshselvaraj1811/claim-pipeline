# Claim Processing Pipeline

A production-ready FastAPI + LangGraph multi-agent service that processes medical insurance claim PDFs. It classifies each page into one of 9 document types and runs specialized extraction agents in parallel.

---

## Architecture

```
POST /api/process
        │
        ▼
┌─────────────────────┐
│   Segregator Agent  │  ← LLM vision: classifies ALL pages into 9 doc types
│   (AI-powered)      │
└────────┬────────────┘
         │  page_map: {doc_type -> [page_numbers]}
         │
    ┌────┴──────────────────────┐
    │    Parallel Extraction     │
    │                            │
    │  ┌──────────┐  ┌────────┐ │  ┌─────────────┐
    │  │ ID Agent │  │Discharge│ │  │ Itemized    │
    │  │          │  │Summary  │ │  │ Bill Agent  │
    │  └──────────┘  └────────┘ │  └─────────────┘
    └────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│     Aggregator      │  ← merges all outputs → final JSON
└─────────────────────┘
         │
         ▼
    JSON Response
```

### Document Types (9)
| Type | Description |
|------|-------------|
| `claim_forms` | Insurance/medical claim forms |
| `cheque_or_bank_details` | Cheques, bank account info |
| `identity_document` | Government IDs, passports |
| `itemized_bill` | Hospital bills with line items |
| `discharge_summary` | Clinical discharge summaries |
| `prescription` | Doctor Rx forms |
| `investigation_report` | Lab reports, blood tests, X-rays |
| `cash_receipt` | Payment receipts |
| `other` | Everything else |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd claim-pipeline
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run

```bash
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

---

## API

### `POST /api/process`

**Request** (multipart/form-data):

| Field | Type | Description |
|-------|------|-------------|
| `claim_id` | string | Unique claim identifier |
| `file` | PDF | The claim PDF to process |

**Example (curl):**

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-2024-789456" \
  -F "file=@final_image_protected.pdf"
```

**Example (Python):**

```python
import requests

with open("final_image_protected.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/process",
        data={"claim_id": "CLM-2024-789456"},
        files={"file": ("claim.pdf", f, "application/pdf")},
    )

print(response.json())
```

**Response:**

```json
{
  "claim_id": "CLM-2024-789456",
  "status": "success",
  "errors": [],
  "page_classification": [
    {"page_number": 1, "doc_type": "claim_forms", "confidence": 0.97, "reasoning": "..."},
    {"page_number": 3, "doc_type": "identity_document", "confidence": 0.99, "reasoning": "..."}
  ],
  "data": {
    "claim_id": "CLM-2024-789456",
    "processing_summary": {
      "total_pages": 18,
      "document_types_found": ["claim_forms", "identity_document", "discharge_summary", "itemized_bill"],
      "agents_executed": ["id_agent", "discharge_summary_agent", "itemized_bill_agent"],
      "errors": []
    },
    "identity": {
      "patient_name": "John Michael Smith",
      "date_of_birth": "March 15, 1985",
      "id_numbers": {"government_id": "ID-987-654-321", "policy_number": "POL-987654321"},
      "policy_details": {"insurance_provider": "HealthCare Insurance Company"},
      "source_pages": [1, 3]
    },
    "discharge_summary": {
      "hospital_name": "City Medical Center",
      "admission_diagnosis": "Community Acquired Pneumonia (CAP)",
      "admission_date": "January 20, 2025",
      "discharge_date": "January 25, 2025",
      "length_of_stay": "5 days",
      "attending_physician": "Dr. Sarah Johnson, MD",
      "discharge_condition": "Stable, improved",
      "discharge_medications": ["Amoxicillin 500mg TID x 7 days", "Acetaminophen 500mg PRN"],
      "source_pages": [4]
    },
    "itemized_bill": {
      "bill_number": "BILL-2025-789456",
      "items": [
        {"description": "Room Charges - Semi-Private", "quantity": 5, "unit_rate": 200.0, "amount": 1000.0},
        {"description": "CT Scan - Chest", "quantity": 1, "unit_rate": 800.0, "amount": 800.0}
      ],
      "total_amount": 6418.65,
      "insurance_payment": 5134.92,
      "patient_responsibility": 1283.73,
      "source_pages": [9]
    }
  }
}
```

---

## Project Structure

```
claim-pipeline/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # Settings (env vars)
│   ├── models.py            # Pydantic state + API models
│   ├── agents/
│   │   ├── segregator.py    # Page classification (LLM vision)
│   │   ├── id_agent.py      # Identity extraction
│   │   ├── discharge_summary_agent.py
│   │   ├── itemized_bill_agent.py
│   │   └── aggregator.py    # Result merging
│   ├── graph/
│   │   └── pipeline.py      # LangGraph StateGraph definition
│   ├── api/
│   │   └── routes.py        # POST /api/process
│   └── utils/
│       ├── llm_client.py    # Anthropic/OpenAI abstraction
│       ├── pdf_utils.py     # PyMuPDF text + image extraction
│       ├── json_parser.py   # Robust JSON extraction from LLM
│       └── logger.py        # Structured logging
├── tests/
│   └── test_pipeline.py     # Unit + integration tests
├── Dockerfile
├── render.yaml              # Render.com deployment
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Deployment

### Docker

```bash
docker build -t claim-pipeline .
docker run -p 8000:8000 --env-file .env claim-pipeline
```

### Render.com

1. Push repo to GitHub
2. Create new Web Service on Render, connect repo
3. Render auto-detects `render.yaml`
4. Set `GROQ_API_KEY` in Render dashboard → Environment

### Railway / Fly.io

```bash
# Railway
railway up

# Fly.io
fly launch
fly secrets set GROQ_API_KEY=gsk_z.....
fly deploy
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `anthropic` or `openai` |
| `GROQ_API_KEY` | — | Required for Anthropic |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Model name |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `PDF_IMAGE_DPI` | `150` | DPI for page rendering |
| `MAX_UPLOAD_SIZE_MB` | `20` | Max PDF upload size |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## LangGraph Flow Detail

```python
START → segregator → extraction_agents (parallel) → aggregator → END
```

- **segregator**: renders all PDF pages as images, sends to LLM in one batch call, gets page-level classifications
- **extraction_agents**: 3 agents run concurrently via `ThreadPoolExecutor`; each receives only its assigned page images/text from the segregator's `page_map`
- **aggregator**: merges all 3 agent outputs into a single structured JSON

---
