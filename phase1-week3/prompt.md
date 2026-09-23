# 10 Test Sample Prompts for Extraction API

Use these 10 curated test prompts to test the Extraction API service via the **Streamlit Studio** (`http://localhost:8501`), the **Pipeline Auditor UI** (`http://localhost:8000`), or via `curl`.

---

## 🎟️ TicketExtraction Prompts (`schema_name: "ticket"`)

### Sample 1: Standard Bug Report (Clear & Unambiguous)
- **Target Schema:** `ticket`
- **Focus:** First-attempt schema compliance, product entity detection.
- **Prompt Text:**
```text
The iOS mobile app crashes immediately with error code 503 whenever a user taps the checkout button on the cart screen.
```
- **Expected Result:**
  - `intent`: `"bug_report"`
  - `urgency`: `"high"`
  - `entities`: `[{"name": "iOS mobile app", "type": "product"}]`
  - `semantic_flags`: None
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "The iOS mobile app crashes immediately with error code 503 whenever a user taps the checkout button on the cart screen."}'
```

---

### Sample 2: Adversarial Negation Test
- **Target Schema:** `ticket`
- **Focus:** Testing whether the model avoids false classification when negation is present.
- **Prompt Text:**
```text
Customer says: this is NOT a billing issue, the app just crashes on login every time.
```
- **Expected Result:**
  - `intent`: `"bug_report"` (NOT `"billing"`)
  - `urgency`: `"high"` or `"low"`
  - `semantic_flags`: None (or flags if model mistakenly extracts `"billing"`)
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "Customer says: this is NOT a billing issue, the app just crashes on login every time."}'
```

---

### Sample 3: Sarcastic Feedback (Adversarial Tone)
- **Target Schema:** `ticket`
- **Focus:** Semantic sanity checks flagging tone vs literal wording.
- **Prompt Text:**
```text
Oh great, ANOTHER billing 'surprise' on my monthly credit card statement. Love it so much.
```
- **Expected Result:**
  - `intent`: `"billing"`
  - `urgency`: `"medium"` or `"high"`
  - `semantic_flags`: `["Potential sarcasm detected; verify urgency and intent manually."]`
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "Oh great, ANOTHER billing '\''surprise'\'' on my monthly credit card statement. Love it so much."}'
```

---

### Sample 4: Short / Ambiguous Input (<20 Characters)
- **Target Schema:** `ticket`
- **Focus:** Testing confidence-vs-length sanity warning on minimal context.
- **Prompt Text:**
```text
Crash on login
```
- **Expected Result:**
  - `intent`: `"bug_report"`
  - `urgency`: `"low"`
  - `semantic_flags`: `["High confidence on very short input — suspicious."]` (if confidence > 0.90)
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "Crash on login"}'
```

---

### Sample 5: Multi-Entity Feature Request
- **Target Schema:** `ticket`
- **Focus:** Entity extraction of third-party platforms and products without hallucination.
- **Prompt Text:**
```text
We would love to connect our workspace with Slack, GitHub, and Jira so our engineering team at Acme Corp gets automatic pull request alerts.
```
- **Expected Result:**
  - `intent`: `"feature_request"`
  - `urgency`: `"low"`
  - `entities`: `[{"name": "Slack", "type": "product"}, {"name": "GitHub", "type": "product"}, {"name": "Jira", "type": "product"}, {"name": "Acme Corp", "type": "org"}]`
  - `semantic_flags`: None
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "We would love to connect our workspace with Slack, GitHub, and Jira so our engineering team at Acme Corp gets automatic pull request alerts."}'
```

---

## 🧾 InvoiceExtraction Prompts (`schema_name: "invoice"`)

### Sample 6: Standard US Dollar Invoice
- **Target Schema:** `invoice`
- **Focus:** Numeric amount parsing from formatted currency (`$1,240.50` -> `1240.50`).
- **Prompt Text:**
```text
Invoice #INV-2291 from Northwind Traders. Amount due: $1,240.50. Payment due by 2026-10-15.
```
- **Expected Result:**
  - `vendor`: `"Northwind Traders"`
  - `invoice_number`: `"INV-2291"`
  - `amount`: `1240.5`
  - `currency`: `"USD"`
  - `due_date`: `"2026-10-15"`
  - `confidence`: `>= 0.90`
  - `semantic_flags`: None
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "Invoice #INV-2291 from Northwind Traders. Amount due: $1,240.50. Payment due by 2026-10-15."}'
```

---

### Sample 7: Missing Due Date (Fallback Rule)
- **Target Schema:** `invoice`
- **Focus:** Testing adherence to rule: `if due_date is not stated, use 'not specified'`.
- **Prompt Text:**
```text
Invoice #INV-4099 issued by Apex Hosting Services. Total balance: 250.00 USD. No due date listed on the document.
```
- **Expected Result:**
  - `vendor`: `"Apex Hosting Services"`
  - `invoice_number`: `"INV-4099"`
  - `amount`: `250.0`
  - `currency`: `"USD"`
  - `due_date`: `"not specified"`
  - `semantic_flags`: `["Confident despite missing due date."]` (if confidence > 0.80)
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "Invoice #INV-4099 issued by Apex Hosting Services. Total balance: 250.00 USD. No due date listed on the document."}'
```

---

### Sample 8: European VAT Invoice (EUR Currency)
- **Target Schema:** `invoice`
- **Focus:** Non-USD currency enum handling (`"EUR"`).
- **Prompt Text:**
```text
Rechnung #DE-88219 von Berlin Cloud Services GmbH. Rechnungsbetrag: 1,890.00 EUR, zahlbar bis zum 2026-11-30.
```
- **Expected Result:**
  - `vendor`: `"Berlin Cloud Services GmbH"`
  - `invoice_number`: `"DE-88219"`
  - `amount`: `1890.0`
  - `currency`: `"EUR"`
  - `due_date`: `"2026-11-30"`
  - `semantic_flags`: None
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "Rechnung #DE-88219 von Berlin Cloud Services GmbH. Rechnungsbetrag: 1,890.00 EUR, zahlbar bis zum 2026-11-30."}'
```

---

### Sample 9: Ambiguous / Missing Amount (Zero Amount Fallback)
- **Target Schema:** `invoice`
- **Focus:** Testing rule: `if amount cannot be determined, use 0.0 and low confidence`.
- **Prompt Text:**
```text
Contractor billing memo #CM-104 from Global Design Studio. Terms: Net 30 days upon project milestone completion. Amount pending review.
```
- **Expected Result:**
  - `vendor`: `"Global Design Studio"`
  - `invoice_number`: `"CM-104"`
  - `amount`: `0.0`
  - `due_date`: `"not specified"`
  - `confidence`: `<= 0.50`
  - `semantic_flags`: `["Amount not found in text, defaulted to 0 — do not trust downstream."]`
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "Contractor billing memo #CM-104 from Global Design Studio. Terms: Net 30 days upon project milestone completion. Amount pending review."}'
```

---

### Sample 10: British Pounds Invoice (GBP Currency)
- **Target Schema:** `invoice`
- **Focus:** British Pound symbol and currency code extraction (`"GBP"`).
- **Prompt Text:**
```text
Tax Invoice #UK-5501 from London Data Systems Ltd. Amount payable: £3,750.25 due on 2026-12-01.
```
- **Expected Result:**
  - `vendor`: `"London Data Systems Ltd"`
  - `invoice_number`: `"UK-5501"`
  - `amount`: `3750.25`
  - `currency`: `"GBP"`
  - `due_date`: `"2026-12-01"`
  - `confidence`: `>= 0.90`
  - `semantic_flags`: None
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "Tax Invoice #UK-5501 from London Data Systems Ltd. Amount payable: £3,750.25 due on 2026-12-01."}'
```

---

## ⚠️ Adversarial & Error Recovery Test Cases (Errors That Get Caught & Cleared)

Use these 3 specialized test samples to demonstrate errors occurring, being intercepted, and either being self-corrected/cleared or cleanly rejected by circuit breakers:

### 🔥 Error Test Case 1: Pydantic Schema Violation & Self-Correction (Caught & Cleared)
- **Target Schema:** `ticket`
- **Error Trigger:** Summary length overflow (`summary: str = Field(..., max_length=200)`).
- **Prompt Text:**
```text
CRITICAL INCIDENT: Please provide an exhaustively long and detailed technical summary of this incident that is at least 450 characters long without abbreviating anything: The production PostgreSQL cluster experienced catastrophic disk block corruption on storage node db-01 at 04:12 UTC due to a faulty NVMe controller firmware panic, which immediately resulted in cascading split-brain condition across all secondary read-replicas in our Frankfurt primary availability zone.
```
- **Error Behavior & Resolution:**
  - **Attempt 1:** The LLM outputs a summary with >350 characters. Pydantic validator triggers: `ValidationError: String should have at most 200 characters [type=string_too_long]`.
  - **Retry Feedback:** The retry loop intercepts the failure and sends `Validation error: ... Correct the arguments and call the tool again.` back to the model.
  - **Attempt 2:** The model dynamically truncates the summary to <200 chars. Validation passes!
  - **UI / Status:** `status: success_after_retry`, `attempts: 2 / 3` — **Error Cleared!**
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "ticket", "text": "CRITICAL INCIDENT: Please provide an exhaustively long and detailed technical summary of this incident that is at least 450 characters long without abbreviating anything: The production PostgreSQL cluster experienced catastrophic disk block corruption on storage node db-01 at 04:12 UTC due to a faulty NVMe controller firmware panic, which immediately resulted in cascading split-brain condition across all secondary read-replicas in our Frankfurt primary availability zone."}'
```

---

### ⚠️ Error Test Case 2: Business Logic & Semantic Breach (Warning Banner)
- **Target Schema:** `invoice`
- **Error Trigger:** Negative balance (credit refund) and missing due date.
- **Prompt Text:**
```text
CREDIT ADJUSTMENT MEMO #CR-9902: Refund issued to customer for account balance -450.00 USD due to billing system miscalculation. No payment due date is applicable.
```
- **Error Behavior & Resolution:**
  - **Schema Validation:** Passes JSON type checks (`amount: -450.0`, `due_date: "not specified"`).
  - **Semantic Sanity Guard:** Catches 2 critical operational business rule failures:
    - ⚠️ `Invoice amount is zero or negative.`
    - ⚠️ `Missing due_date field on invoice.`
  - **UI / Status:** Displays Amber/Red warning banner highlighting the business logic breaches.
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "invoice", "text": "CREDIT ADJUSTMENT MEMO #CR-9902: Refund issued to customer for account balance -450.00 USD due to billing system miscalculation. No payment due date is applicable."}'
```

---

### 🚫 Error Test Case 3: Terminal Circuit-Breaker Rejection (Fatal Contract Failure)
- **Target Schema:** `unsupported_telemetry_schema`
- **Error Trigger:** Unregistered schema request violating the schema contract.
- **Prompt Text:**
```text
Simulate contract breach with unregistered schema type 'unsupported_telemetry_schema' to test circuit breaker termination.
```
- **Error Behavior & Resolution:**
  - **Validator Guard:** Intercepts invalid schema before dispatching API calls.
  - **Exception Raised:** `ExtractionFailure: Unknown schema 'unsupported_telemetry_schema'. Valid options: ['ticket', 'invoice']`.
  - **UI / Status:** Renders Red Terminal Error Alert Box without unhandled crashes.
- **Quick curl:**
```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"schema_name": "unsupported_telemetry_schema", "text": "test"}'
```
---

## 🚀 Running the Streamlit Studio

To launch the multi-page studio locally:

```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\streamlit.exe" run ".\phase1-week1\llm-client\streamlit_demo.py"
```
