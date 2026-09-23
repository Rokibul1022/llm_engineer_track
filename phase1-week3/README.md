# 🛡️ Phase 1 Week 3: Schema-Validated Structured Extraction API Service

> **Production Extraction Pipeline:** Tool-Calling on `openai/gpt-oss-120b`, Bounded Self-Correcting Retries, Semantic Sanity Guards, SSE Streaming, and Multi-Interface Pipeline Auditor Studio.

---

## 📑 Table of Contents
1. [Overview & Engineering Goals](#-overview--engineering-goals)
2. [End-to-End System Architecture](#-end-to-end-system-architecture)
3. [Core Technical Pillars](#-core-technical-pillars)
   - [1. Typed Pydantic Schema Contracts](#1-typed-pydantic-schema-contracts)
   - [2. Bounded Self-Correction Retry Engine](#2-bounded-self-correction-retry-engine)
   - [3. Groq Proxy 400 Fault Interception](#3-groq-proxy-400-fault-interception)
   - [4. Non-LLM Semantic Sanity Guard](#4-non-llm-semantic-sanity-guard)
   - [5. FastAPI Service & SSE Streaming](#5-fastapi-service--sse-streaming)
4. [User Interface Interfaces](#-user-interface-interfaces)
   - [Interactive Streamlit Studio (Week 1 + Week 3 Unified)](#1-interactive-streamlit-studio)
   - [Standalone Pipeline Auditor Web UI](#2-standalone-pipeline-auditor-web-ui)
5. [Adversarial Benchmark & Error-Recovery Test Cases](#-adversarial-benchmark--error-recovery-test-cases)
6. [Empirical Test Report Summary](#-empirical-test-report-summary)
7. [Quickstart & Running Locally](#-quickstart--running-locally)

---

## 🎯 Overview & Engineering Goals

In real-world LLM engineering, unconstrained natural language output is a liability. Downstream microservices, databases, and payment processors require strict, deterministically validated data structures. 

Phase 1 Week 3 builds a production-grade **Structured Extraction API Service** that:
- Binds natural language queries to strictly typed Pydantic v2 schemas (`TicketExtraction` and `InvoiceExtraction`) using Groq's high-speed function calling API (`openai/gpt-oss-120b`).
- Eliminates brittle regular expressions or json-mode parsing by generating dynamic JSON Schemas directly from Pydantic model definitions.
- Enforces an automated **self-correction retry loop** that intercepts schema violations and feeds the exact validation error back to the model for inline correction.
- Implements a deterministic **semantic sanity guard** to catch the failure class that JSON Schema validators cannot see: outputs that are schema-valid but factually or logically wrong (e.g. sarcasm, negation inversion, negative balances).

---

## 🏛️ End-to-End System Architecture

```
                       ┌─────────────────────────────────────────────────┐
                       │          Client / Frontends                     │
                       │  • Streamlit Studio UI (http://localhost:8501)  │
                       │  • Pipeline Auditor UI (http://localhost:8000)  │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                │ HTTP POST /extract  or  SSE /extract/stream
                                                ▼
                       ┌─────────────────────────────────────────────────┐
                       │              FastAPI Gateway (main.py)          │
                       │  • Request ingestion & schema validation        │
                       │  • Serves static/index.html                     │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                       ┌─────────────────────────────────────────────────┐
                       │        ExtractionService (extractor.py)         │
                       │  - Resolves target model in SCHEMA_REGISTRY     │
                       │  - Compiles prompt & dynamic tool spec          │
                       └──────────────┬──────────────────┬───────────────┘
                                      │                  │
                Outbound Tool-Call    │                  │  Attempt 2 Feedback Loop
                {"type": "function"}  │                  │  ("Validation error: ...")
                                      ▼                  │
                       ┌────────────────────────┐        │
                       │ Groq API Client Wrapper│        │
                       │    (groq_client.py)    │        │
                       │  • gpt-oss-120b        │        │
                       │  • 400 Proxy Intercept │        │
                       └──────────────┬─────────┘        │
                                      │                  │
                                      ▼                  │
                       ┌────────────────────────┐        │
                       │ Pydantic v2 Validation ├────────┘
                       │  model_validate(json)  │  (On ValidationError,
                       └──────────────┬─────────┘   retries with feedback)
                                      │
                                      │ Validated Payload
                                      ▼
                       ┌─────────────────────────────────────────────────┐
                       │       Semantic Sanity Guard (semantic_checks)   │
                       │  • Negation inversion detection                 │
                       │  • Sarcasm & sentiment anomaly warnings         │
                       │  • Ungrounded summary detection                 │
                       │  • Invoice zero/negative balance flag           │
                       │  • Missing payment terms / due dates            │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                                    200 OK Validated Response
                                 {"status": "success", ...}
```

---

## 🔬 Core Technical Pillars

### 1. Typed Pydantic Schema Contracts
Located in [`schemas.py`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/schemas.py), extraction structures are modeled as strict Pydantic v2 models:
- **`TicketExtraction`**:
  - `intent`: `Literal["bug_report", "feature_request", "billing", "other"]`
  - `urgency`: `Literal["low", "medium", "high"]`
  - `entities`: `list[Entity]` (typed as person, org, product, location, other)
  - `summary`: String constrained by `Field(..., max_length=200)`
  - `confidence`: `Field(..., ge=0.0, le=1.0)`
- **`InvoiceExtraction`**:
  - `vendor`: String name
  - `invoice_number`: String identifier
  - `amount`: Strict float (parsed from currency strings like `$1,240.50` or `1.890,00 EUR`)
  - `currency`: `Literal["USD", "EUR", "GBP", "BDT", "other"]`
  - `due_date`: ISO format date string, or `"not specified"`
  - `confidence`: `Field(..., ge=0.0, le=1.0)`

### 2. Bounded Self-Correction Retry Engine
In [`extractor.py`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/extractor.py), when the LLM outputs arguments that fail Pydantic validation (e.g. invalid literal enum or summary >200 chars):
1. The error is intercepted: `(json.JSONDecodeError, ValidationError)`.
2. The retry budget is checked (`max_retries=2`).
3. The conversation history is appended with the model's failed tool call, followed by a `tool` role message containing the exact Pydantic error details:
   ```text
   Validation error: 1 validation error for TicketExtraction
   summary: String should have at most 200 characters [type=string_too_long]
   Correct the arguments and call the tool again.
   ```
4. The model corrects its arguments on Attempt 2, resolving the extraction and clearing the error.

### 3. Groq Proxy 400 Fault Interception
In [`groq_client.py`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/groq_client.py), Groq validates tool calls against the JSON Schema at the network gateway and returns an HTTP 400 with `tool_use_failed` if the initial generation fails schema validation. Instead of crashing, the client intercepts the `failed_generation` JSON string, reconstructs a `ToolCall`, and routes it into the Pydantic retry pipeline for automatic self-correction.

### 4. Non-LLM Semantic Sanity Guard
Schema validation only validates structure, not truth. `semantic_checks()` in [`extractor.py`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/extractor.py) scans the output against deterministic heuristics:
- **Negation Inversion**: Catches cases where a user said *"This is NOT a billing issue"* but the LLM classified `intent="billing"`.
- **Sarcasm Detection**: Flags sarcastic tone (e.g. *"Oh great, ANOTHER billing surprise. Love it."*) for manual supervisor review.
- **Short Input Overconfidence**: Flags models claiming `confidence > 0.90` on minimal inputs (`len < 20`).
- **Invoice Logic Flaws**: Flags negative or zero invoice amounts and missing payment due dates.

### 5. FastAPI Service & SSE Streaming
In [`main.py`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/main.py):
- `POST /extract`: Synchronous extraction returning clean data, attempts used, and sanity warnings.
- `POST /extract/stream`: Streams arguments delta chunks as Server-Sent Events (`data: {"partial": "..."}`) and validates the completed buffer once at stream termination (`data: {"final": ...}`).

---

## 🖥️ User Interface Interfaces

### 1. Interactive Streamlit Studio
Accessible at `http://localhost:8501`, fully integrated into the existing LLM Engineer Track workspace:
- **Unified Navigation**: Features Tab 1 (API Extraction), Tab 2 (Live Client & Pipeline), and Tab 3 (Fault Injection).
- **Visual 5-Stage Stepper**: Step 01 Contract ➔ Step 02 Inference ➔ Step 03 Validator ➔ Step 04 Repair ➔ Step 05 Sanity.
- **Metric HUD**: Wall-clock latency, attempts counter, schema compliance badge, and semantic sanity flag count.
- **Attempt History & Self-Correction Audit Box**: Renders the exact Attempt 1 error, the feedback prompt sent to the LLM, and the Attempt 2 recovery confirmation.

### 2. Standalone Pipeline Auditor Web UI
Served directly by FastAPI at `http://localhost:8000/`:
- Live execution pipeline animation matching server response states.
- Real-time Server-Sent Events streaming parameter viewer.
- Audit history ledger with millisecond latency decomposition.

---

## 🧪 Adversarial Benchmark & Error-Recovery Test Cases

All 13 test cases are documented in [`prompt.md`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/prompt.md) and available via the preset dropdown in the UI:

| Preset Name | Target Schema | Behavior / Verification |
|---|---|---|
| **Sample 1: Standard Bug Report** | `ticket` | Direct pass on Attempt 1, entity detection (`iOS mobile app`). |
| **Sample 2: Adversarial Negation** | `ticket` | Verifies intent is `bug_report`, respecting explicit negation (*not billing*). |
| **Sample 3: Sarcastic Feedback** | `ticket` | Flags `Potential sarcasm detected; verify urgency and intent manually.` |
| **Sample 4: Short Ambiguous Input** | `ticket` | Catches high confidence on minimal input (<20 characters). |
| **Sample 5: Multi-Entity Request** | `ticket` | Extracts multiple product and organization entities without hallucination. |
| **Sample 6: Standard Commercial Invoice** | `invoice` | Extracts float amount `$1,240.50` and ISO due date. |
| **Sample 7: Missing Due Date** | `invoice` | Enforces fallback rule `due_date="not specified"`. |
| **Sample 8: European VAT Invoice** | `invoice` | Correctly extracts EUR currency and European date formatting. |
| **Sample 9: Missing / Ambiguous Amount** | `invoice` | Enforces zero amount fallback (`amount=0.0`, low confidence). |
| **Sample 10: British Pounds Invoice** | `invoice` | Extracts GBP currency symbol (`£3,750.25`). |
| **🔥 Error & Cleared: Overlong Summary** | `ticket` | **Causes Attempt 1 ValidationError (>200 chars), self-corrects and clears on Attempt 2.** |
| **⚠️ Error Flagged: Negative Refund Memo** | `invoice` | Valid schema but flags negative balance and missing due date. |
| **🚫 Terminal Error: Unsupported Schema** | `unsupported` | Triggers typed `ExtractionFailure` and displays circuit breaker alert. |

---

## 📊 Empirical Test Report Summary

From [`week3-test-report.md`](file:///c:/Users/hrrok/Desktop/llm_engineer_track/phase1-week3/week3-test-report.md):

| Category | Conditions Evaluated | Pass Rate | Status |
|---|---|:---:|:---:|
| **1. Structural Validation** | Well-formed, Missing fields, Forced malformed enum, Retry exhaustion, Non-tool plain text | **5 / 5** | **100% PASS** |
| **2. Semantic Sanity** | Negation, Sarcasm, Ungrounded summary, Short text overconfidence, Entity hallucination | **5 / 5** | **100% PASS** |
| **3. SSE Streaming** | Partial deltas, Single-pass buffer validation, Mid-stream schema error, Client disconnect | **4 / 4** | **100% PASS** |
| **4. Determinism** | 5x identical runs at $T=0.0$ on `openai/gpt-oss-120b` | **5 / 5** | **100% STABLE** |

---

## 🚀 Quickstart & Running Locally

### 1. Set Up Environment Variables
Create or verify `.env` in `phase1-week3/` or `phase1-week1/llm-client/`:
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 2. Run the FastAPI Backend & Auditor Web UI
```powershell
cd phase1-week3
& "..\phase1-week1\llm-client\.venv\Scripts\uvicorn.exe" main:app --reload --port 8000
```
Open **`http://localhost:8000/`** to view the Pipeline Auditor interface.

### 3. Launch the Unified Streamlit Studio
```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\streamlit.exe" run ".\phase1-week1\llm-client\streamlit_demo.py"
```
Open **`http://localhost:8501/`** and select **Tab 1: API Extraction** to test structured extraction and watch self-correcting retries live!
