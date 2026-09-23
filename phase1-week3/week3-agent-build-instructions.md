# Build Instructions — Extraction API Service (Ticket + Invoice)

Give this document, `week3-build-guide.md`, and `extraction-demo.html` together to whoever/whatever builds this (yourself, or a coding agent). This file is the step-by-step execution order.

---

## 0. Prerequisites

- Python 3.11+
- A Groq API key (`GROQ_API_KEY`), model: `openai/gpt-oss-120b`
- `pip install fastapi uvicorn groq pydantic python-dotenv`

---

## 1. Project structure

```
extraction-service/
├── .env                      # GROQ_API_KEY=... (never commit this)
├── main.py                   # FastAPI app, serves API + the HTML
├── schemas.py                # Pydantic models (TicketExtraction, InvoiceExtraction)
├── prompts.py                # system prompts + tool defs, one pair per schema
├── groq_client.py            # Groq call wrapper (sync + streaming)
├── extractor.py               # ExtractionService: prompt → call → validate → retry
├── exceptions.py             # ExtractionFailure
├── static/
│   └── index.html            # the demo file, rewired (Step 5)
└── requirements.txt
```

---

## 2. `schemas.py` — define both schemas

Port the two field sets straight from the HTML's `SCHEMAS.ticket.jsonSchema` / `SCHEMAS.invoice.jsonSchema` blocks — they're already the spec:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Entity(BaseModel):
    name: str
    type: Literal["person", "org", "product", "location", "other"]

class TicketExtraction(BaseModel):
    intent: Literal["bug_report", "feature_request", "billing", "other"]
    urgency: Literal["low", "medium", "high"]
    entities: list[Entity] = Field(default_factory=list)
    summary: str = Field(..., max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)

class InvoiceExtraction(BaseModel):
    vendor: str
    invoice_number: str
    amount: float
    currency: Literal["USD", "EUR", "GBP", "BDT", "other"]
    due_date: str  # ISO date, or "not specified"
    confidence: float = Field(..., ge=0.0, le=1.0)

SCHEMA_REGISTRY = {
    "ticket": TicketExtraction,
    "invoice": InvoiceExtraction,
}
```

---

## 3. `prompts.py` — one system prompt per schema

Reuse the ticket prompt from the build guide §4 verbatim. Write the matching invoice one the same way:

```python
TICKET_SYSTEM_PROMPT = """You are a structured data extraction engine...
[paste exactly from week3-build-guide.md §4]"""

INVOICE_SYSTEM_PROMPT = """You are a structured data extraction engine. You will be
given invoice text. Your only job is to call the `extract_invoice_data` tool exactly
once with the fields filled in based on the text.

Rules:
- Extract only what is stated in the text. Do not invent amounts, vendors, or dates.
- If `due_date` is not stated, use the literal string "not specified" — do not guess.
- If `amount` cannot be determined, use 0.0 and reflect this in low `confidence`.
- Do not call any tool other than `extract_invoice_data`.
- Do not add commentary before or after the tool call."""

PROMPTS = {"ticket": TICKET_SYSTEM_PROMPT, "invoice": INVOICE_SYSTEM_PROMPT}

def tool_def_for(schema_name: str, model_cls):
    return {
        "type": "function",
        "function": {
            "name": f"extract_{schema_name}_data",
            "description": f"Extract structured fields for {schema_name}.",
            "parameters": model_cls.model_json_schema(),
        },
    }
```

---

## 4. `extractor.py` — the real version of the demo's `mockModelCall` + retry loop

This is a direct translation of the JS `runExtraction()` loop into the actual retry logic from the build guide §5. Port it as-is:

```python
import json
from pydantic import ValidationError
from schemas import SCHEMA_REGISTRY
from prompts import PROMPTS, tool_def_for
from groq_client import call_groq
from exceptions import ExtractionFailure

def extract(schema_name: str, raw_text: str, max_retries: int = 2):
    if schema_name not in SCHEMA_REGISTRY:
        raise ExtractionFailure(f"Unknown schema '{schema_name}'. Valid options: {list(SCHEMA_REGISTRY)}")
    model_cls = SCHEMA_REGISTRY[schema_name]
    tool_def = tool_def_for(schema_name, model_cls)
    messages = [
        {"role": "system", "content": PROMPTS[schema_name]},
        {"role": "user", "content": raw_text},
    ]
    attempts_used = 0
    for attempt in range(max_retries + 1):
        attempts_used += 1
        resp = call_groq(messages, tool_def, tool_name=f"extract_{schema_name}_data")
        tool_call = resp.choices[0].message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            result = model_cls.model_validate(args)
            status = "success" if attempts_used == 1 else "success_after_retry"
            return {"status": status, "attempts": attempts_used, "data": result.model_dump()}
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                raise ExtractionFailure(f"Failed after {attempts_used} attempts: {e}")
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "content": f"Validation error: {e}. Correct the arguments and call the tool again."
            })
```

Add the semantic checks from the build guide §5 (second half) as a separate function called after success — this is what fills the "semantic warnings" box in the UI, same as `SCHEMAS.ticket.flags()` / `SCHEMAS.invoice.flags()` did in JS. Port that logic to Python directly; it's the same conditions.

---

## 5. `main.py` — expose it, and serve the HTML from the same app

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from extractor import extract, semantic_checks
from exceptions import ExtractionFailure

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ExtractRequest(BaseModel):
    schema_name: str  # "ticket" | "invoice"
    text: str

@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    try:
        result = extract(req.schema_name, req.text)
        result["semantic_flags"] = semantic_checks(result["data"], req.text)
        return result
    except ExtractionFailure as e:
        return {"status": "failed", "error": str(e)}

@app.post("/extract/stream")
async def extract_stream_endpoint(req: ExtractRequest):
    if req.schema_name not in SCHEMA_REGISTRY:
        async def bad_schema():
            yield f"data: {json.dumps({'error': f'Unknown schema {req.schema_name!r}'})}\n\n"
        return StreamingResponse(bad_schema(), media_type="text/event-stream")

    model_cls = SCHEMA_REGISTRY[req.schema_name]
    tool_def = tool_def_for(req.schema_name, model_cls)
    tool_name = f"extract_{req.schema_name}_data"
    messages = [
        {"role": "system", "content": PROMPTS[req.schema_name]},
        {"role": "user", "content": req.text},
    ]

    async def stream_extraction():
        stream = call_groq(messages, tool_def, tool_name=tool_name, stream=True)
        buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                arg_piece = delta.tool_calls[0].function.arguments or ""
                buffer += arg_piece
                yield f"data: {json.dumps({'partial': arg_piece})}\n\n"
        try:
            parsed = model_cls.model_validate(json.loads(buffer))
            flags = semantic_checks(parsed.model_dump(), req.text)
            yield f"data: {json.dumps({'final': parsed.model_dump(), 'semantic_flags': flags})}\n\n"
        except (json.JSONDecodeError, ValidationError) as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_extraction(), media_type="text/event-stream")
```

Note the two imports this adds at the top of `main.py`: `import json` and `from pydantic import ValidationError` (alongside `tool_def_for` and `PROMPTS` from `prompts.py`). This version has no retry — matching the build guide §6, streaming validates the final assembled object once and reports success or failure; it doesn't re-prompt mid-stream. If you want retry-on-stream too, that's a deliberate scope decision to make explicitly, not a default to reach for.

Serving the HTML from the same FastAPI process (via `StaticFiles`, or a plain `@app.get("/")` returning the file) avoids CORS entirely — the page and the API share an origin.

---

## 6. Rewire `static/index.html` (the demo file)

Three edits to the existing HTML/JS, nothing else changes — the pipeline visualizer, audit trail, and field rendering all stay exactly as designed:

1. **Delete** the "Force"/"Random" `<select id="failureMode">` block entirely, and every reference to `mode` in `mockModelCall`. Those were demo-only fault injection — production has no such knob.
2. **Replace** `mockModelCall(schema, text, mode, attempt)` with a real call:
   ```js
   async function callExtractAPI(schemaKey, text) {
     const resp = await fetch("/extract", {
       method: "POST",
       headers: {"Content-Type": "application/json"},
       body: JSON.stringify({schema_name: schemaKey, text})
     });
     return await resp.json();
   }
   ```
   The retry loop currently simulated in JS (`while(attempt <= maxAttempts)`) is now happening server-side inside `extract()` — the frontend just calls `/extract` once and gets back `{status, attempts, data, semantic_flags}`. Simplify the JS loop to a single call, and drive the step-tracker animation off the returned `attempts` count and `status` value instead of simulating it turn-by-turn.
3. **For the streaming variant**, use `EventSource` or a `fetch` + `ReadableStream` reader against `/extract/stream`, updating the pipeline steps as `partial`/`final`/`error` SSE events arrive — mirroring the build guide §6 client pattern.

---

## 7. Run it

```bash
# .env
GROQ_API_KEY=your_key_here
```

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/` — you're now looking at the real pipeline, not the simulation.

---

## 8. Verify against the test report

Run through `week3-test-report.md` against the live service, not the mock. The structural cases (malformed JSON, missing field, wrong enum) will no longer be forceable by a dropdown — instead, prompt the model with genuinely ambiguous/edge-case text and observe whether real failures happen and get handled the same way the demo showed. Fill in the `Result` columns with what actually happens.

---

## Order of operations, if handing this to an agent in one shot

1. Build `schemas.py`, `prompts.py`, `exceptions.py` (no API calls, pure logic — testable immediately).
2. Build `groq_client.py` + `extractor.py`, test against real Groq with a couple of manual `curl` calls to a temporary `/extract` endpoint.
3. Wire `main.py` fully, confirm `/extract` and `/extract/stream` work via `curl`/Postman before touching the frontend.
4. Only then do Step 6 (rewire the HTML) — the backend should be proven correct on its own first, so any bug you see in the UI is a frontend bug, not an ambiguous "which layer broke" situation.
5. Run the test report last, against the fully wired system.
