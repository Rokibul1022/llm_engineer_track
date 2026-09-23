"""FastAPI service exposing synchronous extraction and SSE streaming endpoints."""
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from exceptions import ExtractionFailure
from extractor import extract, semantic_checks
from groq_client import call_groq
from prompts import PROMPTS, tool_def_for
from schemas import SCHEMA_REGISTRY

app = FastAPI(title="Schema-Validated Extraction Service")

# Allow CORS for flexible local development and preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ExtractRequest(BaseModel):
    schema_name: str  # "ticket" | "invoice"
    text: str
    force_initial_fault: bool = False


@app.get("/")
def index():
    """Serve the auditor view HTML page."""
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Extraction API running. Please visit /static/index.html"}


@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    """Synchronous extraction with self-correcting retry loop and semantic sanity checks."""
    try:
        result = extract(req.schema_name, req.text, force_initial_fault=req.force_initial_fault)
        result["semantic_flags"] = semantic_checks(result["data"], req.text)
        return result
    except ExtractionFailure as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": f"Internal error: {e}"}


@app.post("/extract/stream")
async def extract_stream_endpoint(req: ExtractRequest):
    """Server-Sent Events streaming endpoint emitting partial deltas and a final validated object."""
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
        try:
            stream = call_groq(messages, tool_def, tool_name=tool_name, stream=True)
            buffer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    arg_piece = delta.tool_calls[0].function.arguments or ""
                    buffer += arg_piece
                    yield f"data: {json.dumps({'partial': arg_piece})}\n\n"

            # Final validation once stream completes
            try:
                parsed = model_cls.model_validate(json.loads(buffer))
                flags = semantic_checks(parsed.model_dump(), req.text)
                yield f"data: {json.dumps({'final': parsed.model_dump(), 'semantic_flags': flags})}\n\n"
            except (json.JSONDecodeError, ValidationError) as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_extraction(), media_type="text/event-stream")
