"""FastAPI wrapper around AsyncLLMClient.

One endpoint, one job: accept a validated LLMRequest, return a
validated LLMResponse, and translate internal exceptions into the
right HTTP status codes instead of leaking a 500 for everything.
"""
import json
import os
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from llm_client import (
    AsyncLLMClient,
    LLMBadRequestError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)

client: AsyncLLMClient | None = None

# Defaults target Groq's OpenAI-compatible endpoint with the free
# openai/gpt-oss-120b model. Override via env if you switch providers —
# nothing provider-specific is hardcoded beyond these two defaults.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY environment variable is missing! "
            "Please set LLM_API_KEY in your .env file or environment."
        )
    client = AsyncLLMClient(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
    )
    yield
    await client.aclose()


app = FastAPI(title="TDI LLM Client Service", lifespan=lifespan)

# Local-demo only: lets the static page on the same machine call the API
# from the browser. Tighten this before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/complete", response_model=LLMResponse)
async def complete(request: LLMRequest) -> LLMResponse:
    assert client is not None
    try:
        return await client.complete(request)
    except LLMBadRequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (LLMProviderError, LLMMalformedResponseError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/stream")
async def stream(request: LLMRequest) -> StreamingResponse:
    """Server-Sent Events: one 'data: {"token": ...}' line per token,
    one final 'data: {"usage": ...}' line once the provider reports
    token counts, then 'data: [DONE]'. Kept intentionally simple — the
    demo UI just reads these lines directly rather than needing a full
    SSE client lib."""
    assert client is not None

    async def event_source():
        usage_holder: dict = {}
        try:
            async for token in client.stream(request, on_usage=lambda u: usage_holder.update(u.model_dump())):
                yield f"data: {json.dumps({'token': token})}\n\n"
            if usage_holder:
                yield f"data: {json.dumps({'usage': usage_holder, 'model': request.model})}\n\n"
            yield "data: [DONE]\n\n"
        except LLMBadRequestError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except (LLMRateLimitError, LLMTimeoutError, LLMProviderError, LLMMalformedResponseError) as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
async def config() -> dict[str, str]:
    """Lets the demo UI prefill the right model/base URL without
    hardcoding provider details in two places."""
    return {
        "default_model": os.environ.get("LLM_DEFAULT_MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
    }


# ---- Acceptance Test Simulation / Fault-Injection Endpoint ----
from pydantic import BaseModel, Field
import httpx
import asyncio
import time
from llm_client import ChatMessage
from llm_client.schemas import AttemptLog

SIM_SCRIPTS = {
    "provider_500": ["500", "500", "ok"],
    "rate_limit_429": ["429", "429", "429", "429", "429"],
    "timeout": ["timeout", "timeout", "timeout", "timeout", "timeout"],
    "malformed_200": ["malformed"],
    "bad_request_400": ["400"],
    "success": ["ok"],
}


class SimulationRequest(BaseModel):
    scenario: str
    max_retries: int = Field(default=3, ge=1, le=5)
    base_delay_s: float = Field(default=0.2, ge=0.01, le=2.0)


def _make_mock_transport(script: list[str]) -> httpx.MockTransport:
    remaining = list(script)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = remaining.pop(0) if remaining else "500"
        if outcome == "ok":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "This is a normal, successful simulated response."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            })
        if outcome == "500":
            return httpx.Response(500, text="internal server error")
        if outcome == "429":
            return httpx.Response(429, text="rate limit exceeded")
        if outcome == "400":
            return httpx.Response(400, text="invalid request: unknown model")
        if outcome == "malformed":
            return httpx.Response(200, json={"unexpected": "shape, missing choices/usage"})
        if outcome == "timeout":
            time.sleep(0.25)  # brief realistic delay so timeout wait is observable
            raise httpx.TimeoutException("simulated connection timeout")
        raise ValueError(f"unknown outcome: {outcome}")

    return httpx.MockTransport(handler)


@app.post("/v1/simulate")
async def simulate(req: SimulationRequest) -> StreamingResponse:
    """Streams live attempt logs as the client executes against the simulated
    fault-injection transport. Yields real-time events for each attempt, its backoff
    delay, and the final response or terminal error."""
    script = SIM_SCRIPTS.get(req.scenario, ["ok"])
    transport = _make_mock_transport(script)
    sim_client = AsyncLLMClient(
        api_key="demo-key",
        max_retries=req.max_retries,
        base_delay_s=req.base_delay_s,
        max_delay_s=2.0,
        transport=transport,
    )
    test_request = LLMRequest(model="gpt-sim", messages=[ChatMessage(role="user", content="hello")])

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        # Emit initial simulation metadata
        await queue.put({
            "sim_start": {
                "scenario": req.scenario,
                "max_retries": req.max_retries,
                "base_delay_s": req.base_delay_s,
            }
        })

        def on_attempt(log: AttemptLog) -> None:
            queue.put_nowait(log)

        async def worker():
            try:
                res = await sim_client.complete(test_request, on_attempt=on_attempt)
                await queue.put({"final_success": True, "data": res.model_dump()})
            except Exception as exc:
                await queue.put({
                    "final_error": True,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                })
            finally:
                await sim_client.aclose()
                await queue.put(None)

        task = asyncio.create_task(worker())
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, AttemptLog):
                yield f"data: {json.dumps({'attempt': item.model_dump()})}\n\n"
            else:
                yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"
        await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serves static/index.html at "/" so the demo UI is available the moment
# the server is running, with no separate process needed.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")

