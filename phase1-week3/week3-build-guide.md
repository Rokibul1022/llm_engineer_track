# Week 3 Build Guide — Schema-Validated Extraction API

**Phase deliverable:** LLM API service
**Model:** `openai/gpt-oss-120b` via Groq
**Goal:** A FastAPI service that extracts structured data from unstructured text using tool calling, validates it against a JSON Schema, handles failures gracefully, and can stream.

---

## 1. What "done" looks like

- `POST /extract` — takes raw text + (optionally) a target schema name, returns validated JSON.
- `POST /extract/stream` — same thing, but streams tokens via SSE and emits the final validated object at the end.
- Output is **guaranteed** to match a Pydantic schema, or the endpoint returns a structured error — never silently returns malformed data.
- Retries on schema-invalid output, with a capped retry budget.
- Logs both classes of failure: *structural* (bad JSON / schema mismatch) and *semantic* (valid JSON, wrong content) — the second one you check with spot rules, not with hope.

---

## 2. Architecture

```
                       ┌─────────────────────────┐
 Client ── POST ──────▶│      FastAPI app        │
                       │  /extract  /extract/stream │
                       └────────────┬────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   ExtractionService   │
                         │  (orchestration layer)│
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼───────────────────┐
                 ▼                  ▼                    ▼
        ┌────────────────┐ ┌───────────────┐   ┌──────────────────┐
        │ Prompt Builder  │ │ Groq Client    │   │ Schema Registry   │
        │ (system+tool    │ │ (tool calling, │   │ (Pydantic models  │
        │  definitions)   │ │  streaming)    │   │  + JSON Schemas)  │
        └────────────────┘ └───────┬───────┘   └──────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Validator/Repair    │
                         │  - pydantic.validate  │
                         │  - retry w/ error msg │
                         │  - fallback strategy  │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Response / SSE out  │
                         └───────────────────────┘
```

**Layers, and why each exists:**

- **Schema Registry** — one Pydantic model per extraction task (e.g. `TicketExtraction`, `InvoiceExtraction`). Pydantic gives you the JSON Schema for free (`Model.model_json_schema()`), which you feed straight into the tool definition — the schema is the single source of truth for both the tool spec and the validator.
- **Prompt Builder** — constructs the system prompt + tool definition per request. Keep this separate from the Groq client so you can unit-test prompt construction without hitting the API.
- **Groq Client wrapper** — thin wrapper around the Groq SDK (OpenAI-compatible) that forces `tool_choice` to your extraction tool, handles streaming vs non-streaming, and normalizes Groq's response shape.
- **Validator/Repair** — the part everyone skips and shouldn't. Structural failures (bad JSON, missing required field) get one automatic repair attempt by re-prompting with the validation error. Semantic checks (does `email` field look like an email, is `date` parseable, is `confidence` in range) run as a second pass with plain Python, not the LLM.

---

## 3. Schema definition (source of truth)

```python
# schemas.py
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
```

`TicketExtraction.model_json_schema()` gives you the schema to hand to the model as a tool. Don't hand-write a separate JSON Schema — it will drift from the Pydantic model within a week.

---

## 4. Prompt design

The core move: **don't ask the model to "output JSON."** Force it through Groq's tool-calling interface, so the model's only valid move is to call a function with arguments matching your schema. This eliminates most of the "here's your JSON wrapped in a sentence and markdown fences" failure mode.

### System prompt

```
You are a structured data extraction engine. You will be given a piece of
raw text. Your only job is to call the `extract_ticket_data` tool exactly
once with the fields filled in based on the text.

Rules:
- Extract only what is stated or strongly implied in the text. Do not
  invent entities, dates, or values that are not present.
- If a field cannot be determined from the text, use the most conservative
  valid value allowed by its type (e.g. urgency="low" if unstated) and
  reflect your uncertainty in `confidence`.
- `confidence` reflects how well-supported the extraction is by the text,
  not how well-formed the JSON is.
- Do not call any tool other than `extract_ticket_data`.
- Do not add commentary before or after the tool call.
```

Notice the prompt explicitly separates "is this well-formed" from "is this well-supported by the text" — that's you baking the valid-JSON-vs-semantically-correct distinction into the model's instructions, not just your validation code.

### Tool definition (Groq/OpenAI function-calling format)

```python
tool_def = {
    "type": "function",
    "function": {
        "name": "extract_ticket_data",
        "description": "Extract structured fields from a support ticket.",
        "parameters": TicketExtraction.model_json_schema()
    }
}
```

### Call pattern

```python
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ],
    tools=[tool_def],
    tool_choice={"type": "function", "function": {"name": "extract_ticket_data"}},
    temperature=0,  # extraction wants determinism, not creativity
)
```

Forcing `tool_choice` to the specific function (rather than `"auto"`) removes the failure mode where the model responds in plain text instead of calling the tool.

---

## 5. Validation and repair loop

```python
import json
from pydantic import ValidationError

def extract_with_retry(raw_text: str, max_retries: int = 2) -> TicketExtraction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    for attempt in range(max_retries + 1):
        resp = call_groq(messages, tool_def)
        tool_call = resp.choices[0].message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            return TicketExtraction.model_validate(args)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                raise ExtractionFailure(f"Failed after {max_retries} retries: {e}")
            # feed the error back so the model can self-correct
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Validation error: {e}. Correct the arguments and call the tool again."
            })
```

This is the "failure handling" part of the assignment: structural failures get a bounded number of self-correction turns using the actual Pydantic error message (much more effective than a generic "try again"), then a hard failure with a typed exception rather than an infinite loop or silent bad data.

### Semantic sanity pass (separate from schema validation)

After structural validation succeeds, run a few cheap non-LLM checks — these catch "valid JSON, wrong content":

```python
def semantic_checks(result: TicketExtraction, raw_text: str) -> list[str]:
    warnings = []
    if result.confidence > 0.9 and len(raw_text) < 20:
        warnings.append("High confidence on very short input — suspicious.")
    if result.summary.lower() in raw_text.lower():
        pass  # fine, summary is grounded
    elif len(result.summary) > 0 and not any(w in raw_text.lower() for w in result.summary.lower().split()[:3]):
        warnings.append("Summary may not be grounded in source text.")
    return warnings
```

These don't block the response — they get logged/attached as metadata, since this is exactly the class of error schema validation cannot see.

---

## 6. Streaming endpoint

Groq supports streaming, but tool-call arguments arrive as accumulating deltas, not full text tokens the way normal chat completions do. So: stream to the client for UX responsiveness, but you still validate the *assembled* final object before calling it done.

```python
from fastapi.responses import StreamingResponse
import json

async def stream_extraction(raw_text: str):
    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[...],
        tools=[tool_def],
        tool_choice={"type": "function", "function": {"name": "extract_ticket_data"}},
        stream=True,
    )
    buffer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            arg_piece = delta.tool_calls[0].function.arguments or ""
            buffer += arg_piece
            yield f"data: {json.dumps({'partial': arg_piece})}\n\n"
    # final validation once the stream closes
    try:
        parsed = TicketExtraction.model_validate(json.loads(buffer))
        yield f"data: {json.dumps({'final': parsed.model_dump()})}\n\n"
    except (json.JSONDecodeError, ValidationError) as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/extract/stream")
async def extract_stream_endpoint(text: str):
    return StreamingResponse(stream_extraction(text), media_type="text/event-stream")
```

Client sees partial JSON arriving live, then a `final` event once it's been schema-validated — so the UI can render "extracting..." and then swap to the confirmed structured object.

---

## 7. Suggested project layout

```
extraction-service/
├── main.py              # FastAPI app, routes
├── schemas.py           # Pydantic models = schema registry
├── prompts.py           # system prompt templates per schema
├── groq_client.py       # thin wrapper: call, stream, tool_choice handling
├── extractor.py         # ExtractionService: orchestrates prompt+call+validate+retry
├── exceptions.py        # ExtractionFailure, SchemaMismatch, etc.
├── tests/
│   ├── test_schemas.py
│   ├── test_extractor.py     # mock Groq responses, test retry logic
│   └── fixtures/              # sample tickets + expected extractions
└── requirements.txt
```

---

## 8. Knowledge targets, answered directly

**Prompting vs RAG vs fine-tuning vs pretraining**
- *Pretraining* — the model learns general language/world knowledge from massive raw text; this happened before you ever touched the model, you don't control it.
- *Fine-tuning* — you update the model's weights on task-specific examples, changing its behavior permanently for that checkpoint. You're not doing this — GPT-OSS-120B via Groq is a fixed checkpoint you call over an API.
- *RAG* — you leave the weights alone but inject relevant retrieved documents into the prompt at inference time, so the model has access to information it wasn't trained on or needs freshly. Not used this week — there's no retrieval step, just the raw input text.
- *Prompting* (what you're doing) — you shape output purely through instructions, examples, and structural constraints (tool schemas) given at inference time, with a fixed, untouched model. This week is a pure demonstration that careful prompting + tool calling + validation can get reliable structured output without touching weights or adding retrieval.

**Why valid JSON ≠ semantically correct**
JSON Schema validation checks *shape*: right keys, right types, values within declared ranges/enums. It says nothing about whether `intent: "billing"` is actually the correct intent for the ticket, or whether the model quietly invented an entity that never appeared in the text. A model under tool-calling constraints will almost always produce syntactically valid arguments — that's exactly why it's tempting to treat validation as the finish line. It isn't. That's why this guide has both a structural retry loop (Section 5, first half) and a separate semantic sanity pass (Section 5, second half) — they catch different failure classes and one cannot substitute for the other.

---

## 9. Minimal test plan

- **Structural**: feed intentionally ambiguous/short text, confirm retries fire and eventually either succeed or raise `ExtractionFailure` cleanly (no unhandled exception, no infinite loop).
- **Semantic**: feed text where the "obvious" extraction is wrong (e.g. sarcasm, negation — "this is NOT a billing issue") and check whether `semantic_checks` or manual review catches it; this is where you'll find gpt-oss-120b's actual failure modes.
- **Streaming**: confirm partial chunks arrive before the `final` event, and that a mid-stream schema failure still surfaces as an `error` event rather than hanging the connection.
- **Determinism**: run the same input 5x at `temperature=0`, confirm outputs are stable — if they're not, that's worth noting in your writeup as a limitation of tool-calling determinism on this backend.
