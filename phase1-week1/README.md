# llm-client

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.10+-e92063.svg)](https://docs.pydantic.dev)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg)](tests/test_client.py)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)

A production-grade, typed, asynchronous Python client for calling Large Language Model (LLM) providers reliably, paired with a Dockerized FastAPI proxy service, comprehensive test suite, and an interactive real-time visual demonstration studio.

Built for **Phase 1 / Week 1 of the 24-Week LLM Engineer Track**.

---

## Table of Contents
1. [Core Capabilities](#core-capabilities)
2. [System Architecture](#system-architecture)
3. [Failure-Handling & Distributed Systems Design](#failure-handling--distributed-systems-design)
4. [Latency & Cost Decomposition](#latency--cost-decomposition)
5. [Interactive Visual Demonstration Studio](#interactive-visual-demonstration-studio)
6. [Quickstart & Setup](#quickstart--setup)
7. [Programmatic Library Usage](#programmatic-library-usage)
8. [API Reference & Service Endpoints](#api-reference--service-endpoints)
9. [Acceptance Test Matrix](#acceptance-test-matrix)
10. [Docker Deployment](#docker-deployment)

---

## Core Capabilities

- **Strict Type Safety**: Fully validated request and response payloads using Pydantic v2 (`LLMRequest`, `LLMResponse`, `Usage`).
- **Decoupled Architecture**: `AsyncLLMClient` is a pure async Python package with zero framework coupling. It can be embedded in CLIs, workers, or microservices independently of FastAPI.
- **Predictable Retry Engine**:
  - Automatically retries transient faults: `429 Rate Limit` and `5xx Provider Error`.
  - Fast-fails non-retryable faults: `4xx Bad Request` and `200 Malformed JSON` are terminated immediately on Attempt 1 with zero retry quota waste.
- **Full-Jitter Exponential Backoff**: Uses the decorrelated random full-jitter algorithm ($U(0, \min(\text{cap}, \text{base} \times 2^{\text{attempt}}))$) to prevent thundering herd retry storms.
- **Streaming with Handshake-Only Retries**: Implements Server-Sent Events (SSE) streaming. Retries only apply to initial connection establishment; mid-stream drops are surfaced immediately to prevent token duplication.
- **Complete Observability & Telemetry**: Every call measures monotonic wall-clock latency (`time.perf_counter()`), exact attempt counts, Time-to-First-Token (TTFT), autoregressive decode tokens/sec, and granular API cost breakdown.

---

## System Architecture

```
                                      +------------------------------------+
                                      |          Browser / Client          |
                                      +-----------------+------------------+
                                                        |
                                                        | JSON / SSE Stream
                                                        v
+--------------------------------------------------------------------------------------------------+
| FastAPI Gateway (app.py)                                                                         |
|  - Validates request payload against Pydantic LLMRequest                                         |
|  - Translates domain exceptions -> standard HTTP status codes (400->422, 429->429, 500->502)    |
|  - Streams Server-Sent Events (SSE) chunk-by-chunk                                               |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Calls client.complete() / stream()
                                                        v
+--------------------------------------------------------------------------------------------------+
| AsyncLLMClient Core Engine (llm_client/client.py)                                                |
|  - Monotonic Timer Start (t0 = time.perf_counter())                                              |
|  - httpx.AsyncClient connection pool with Bearer authorization & timeout deadline                |
|                                                                                                  |
|       [ HTTP POST /chat/completions ]                                                            |
|                    |                                                                             |
|                    v                                                                             |
|        +-----------------------+                                                                 |
|        | HTTP Status Evaluator |                                                                 |
|        +-----------+-----------+                                                                 |
|                    |                                                                             |
|         +----------+----------+--------------------+---------------------+                       |
|         | 200 OK              | 429 / 5xx          | 4xx (e.g. 400, 404) | 200 Malformed Body  |
|         v                     v                    v                     v                       |
|    Validate Schema       Compute Backoff      FAST-FAIL              FAST-FAIL                   |
|    Usage + LLMResponse   Sleep (Full Jitter)  LLMBadRequestError     LLMMalformedResponseError   |
|    Return Result         Next Attempt         (Zero retries)         (Zero retries)              |
+--------------------------------------------------------------------------------------------------+
                                                        |
                                                        | HTTPS (Bearer auth)
                                                        v
                                      +------------------------------------+
                                      |    LLM Provider (Groq / OpenAI)    |
                                      +------------------------------------+
```

### Module Structure
```
llm_client/
  ├── __init__.py       # Clean package exports (AsyncLLMClient, Schemas, Exceptions)
  ├── schemas.py        # Pydantic v2 data models: LLMRequest, LLMResponse, Usage, AttemptLog
  ├── exceptions.py     # Domain-specific exception hierarchy
  └── client.py         # AsyncLLMClient implementation (retries, jitter, timing, streaming)
app.py                  # Production FastAPI service and SSE streaming gateway
static/
  └── index.html        # Interactive visual demo, Latency & Cost Decomposition, & Simulation HUD
tests/
  └── test_client.py    # 100% Mocked HTTP test suite using respx (zero live API calls in CI)
Dockerfile              # Multi-stage production container build
pyproject.toml          # Package metadata, dependencies, and build configuration
```

---

## Failure-Handling & Distributed Systems Design

### 1. The Retry Policy Matrix
| Scenario | Status Code | Policy | Rationale | Client Exception | HTTP Status |
|---|:---:|:---:|---|---|:---:|
| **Rate Limit** | `429` | 🔁 **Retry** | Transient quota throttle; recovers with backoff. | `LLMRateLimitError` | `429` |
| **Provider Crash** | `500, 502, 503, 504` | 🔁 **Retry** | Upstream backend failure; retryable on different node. | `LLMProviderError` | `502` |
| **Connection Timeout** | `Timeout` | 🔁 **Retry** | Dropped packet or socket timeout; safe to retry. | `LLMTimeoutError` | `504` |
| **Bad Request** | `400, 401, 404` | 🚫 **Fast-Fail** | Client error (invalid model, malformed body, invalid auth). Retrying repeats the error and wastes rate limit. | `LLMBadRequestError` | `422` |
| **Malformed Response**| `200 OK` (Bad JSON) | 🚫 **Fast-Fail** | Upstream contract breach; output failed Pydantic schema validation. Retrying will produce identical bad output. | `LLMMalformedResponseError` | `502` |

### 2. Backoff Algorithm: Exponential with Full Jitter
Standard exponential backoff causes **thundering herds** when multiple concurrent callers encounter a rate limit at time $t$ and all retry at $t + 2^k$.

This client implements **Full Jitter** as recommended by AWS Distributed Systems Architecture:

$$\text{Sleep Delay} = \text{Uniform}\left(0, \, \min\left(\text{MaxDelay}, \, \text{BaseDelay} \times 2^{\text{attempt}}\right)\right)$$

```python
def _backoff_delay(self, attempt: int) -> float:
    cap = min(self._max_delay_s, self._base_delay_s * (2 ** attempt))
    return random.uniform(0, cap)
```
This distributes retry attempts evenly across the time window, collapsing traffic spikes into a smooth, sustainable arrival rate.

### 3. Idempotency & Retry Storm Mitigation
- Pure chat completions without tools are idempotent from the client's perspective, but **expensive** on the provider side.
- A client-side timeout does **not** guarantee that the LLM provider stopped generation. If the provider continues generating tokens after the client disconnects, blind retries amplify server load (a "retry storm").
- This client mitigates retry storms by enforcing a strict `max_retries` ceiling (default 3), bounded delay caps (`max_delay_s=8.0`), and distinct attempt logging for observability.

### 4. Streaming Safety: Connection Handshake vs. Mid-Stream Drops
- In streaming mode (`/v1/stream`), retries **only** apply while establishing the HTTP connection.
- Once the first token is emitted and read by the consumer, **mid-stream drops are surfaced immediately as exceptions**. Silently restarting generation would duplicate tokens that have already been rendered to the user.

---

## Latency & Cost Decomposition

### Latency Phases
Total round-trip time for an LLM request is broken down into two distinct physical phases:

$$\text{Total Latency} = \text{Phase 1 (Prefill / TTFT)} + \text{Phase 2 (Decode / Generation)}$$

1. **Phase 1: Prefill & Time to First Token (TTFT)**
   - Includes client serialization, TLS handshake, network wire transit, prompt tokenization, and provider KV-cache allocation.
   - For an 80-token prompt, TTFT typically accounts for 60–80% of perceived latency on high-throughput providers.
2. **Phase 2: Autoregressive Decode (Generation)**
   - Autoregressively generates output tokens one-by-one:
     $$\text{Decode Duration} = \frac{\text{Completion Tokens}}{\text{Generation Speed (tokens/s)}}$$
   - At 600 tokens/s, each token requires ~1.6 ms.

### API Billing & Cost Decomposition
The service tracks token costs deterministically using a decoupled input/output pricing model:

$$\text{Total Cost} = \left(\text{Prompt Tokens} \times \frac{\$0.10}{1,000,000}\right) + \left(\text{Completion Tokens} \times \frac{\$0.40}{1,000,000}\right)$$

- **Input Rate**: **`$0.10` per 1 Million tokens** (`$0.00010` per 1K tokens)
- **Output Rate**: **`$0.40` per 1 Million tokens** (`$0.00040` per 1K tokens)
- Granular breakdown is displayed in both the API response metadata and the visual decomposition dashboard.

---

## Interactive Visual Demonstration Studio

The service includes a web interface hosted at `http://localhost:8000` containing two visual demonstration environments:

### Tab 1: ⚡ Live Client, Speed Insights & Execution Pipeline
- **Backend Execution Pipeline**: An animated 5-stage trace showing real-time state progression:
  `[01. Request Ingestion]` ➔ `[02. Async Dispatch]` ➔ `[03. Prefill & TTFT]` ➔ `[04. Autoregressive Decode]` ➔ `[05. Response & Billing]`.
- **Speed Insights HUD**: Live metrics displaying input/output tokens, total inference time, generation speed (tokens/sec), and round-trip duration.
- **Latency & Cost Decomposition Card**: Visual segmented progress bar, dual phase cards (Prefill vs. Decode), and cost formula chips.
- **📐 Trace Spec Drawer**: Expandable architectural view mapping each pipeline step to its underlying Python call stack.

### Tab 2: 🧪 Acceptance Test & Fault Injection Studio
- **6 Failure Scenarios**: Single-click simulation of `500 Provider Error`, `429 Rate Limit`, `Timeout`, `400 Bad Request`, `Malformed Output (200)`, and `Normal 200 OK`.
- **Live Animated Execution Graph**: Dynamically renders execution nodes, state transitions, backoff bars with countdown timers, and terminal resolution states.
- **Policy Decision HUD**: Displays live evaluation state (`IDLE` ➔ `EVALUATING` ➔ `BACKING_OFF` ➔ `RECOVERED / FAST_FAIL`).
- **Verdict Banners**: Green recovery banners highlighting backoff jitter, red fast-fail banners proving zero retry waste.

---

## Quickstart & Setup

### Prerequisites
- Python 3.10 or higher
- An API key from Groq or any OpenAI-compatible provider

### 1. Installation

Clone the repository and install dependencies:

```bash
cd phase1-week1/llm-client

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with all dependencies
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file in the project directory:

```bash
# Groq (default free model: openai/gpt-oss-120b)
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1

# Optional: override default model
# LLM_DEFAULT_MODEL=openai/gpt-oss-120b
```

### 3. Run the Test Suite

The test suite runs against mocked HTTP transports using `respx`. No network access or API key is required:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_client.py::test_complete_success PASSED                         [ 12%]
tests/test_client.py::test_complete_retries_transient_500_then_succeeds PASSED [ 25%]
tests/test_client.py::test_complete_retries_429_exhausts_and_raises PASSED [ 37%]
tests/test_client.py::test_complete_400_fails_fast_no_retry PASSED        [ 50%]
tests/test_client.py::test_complete_timeout_retries_and_exhausts PASSED    [ 62%]
tests/test_client.py::test_complete_malformed_response_fails_fast PASSED   [ 75%]
tests/test_client.py::test_stream_yields_tokens PASSED                     [ 87%]
tests/test_client.py::test_stream_retries_connection_error PASSED          [100%]

============================== 8 passed in 0.42s ==============================
```

### 4. Launch the Local Service

Start the FastAPI application with auto-reload:

```bash
uvicorn app:app --reload --port 8000
```

Open your browser at **`http://localhost:8000`** to access the interactive demonstration studio.

---

## Programmatic Library Usage

The client is completely decoupled from FastAPI and can be used directly in any Python application:

### Non-Streaming Completion

```python
import asyncio
import os
from llm_client import AsyncLLMClient, LLMRequest, LLMBadRequestError, LLMRateLimitError

async def main():
    async with AsyncLLMClient(
        api_key=os.environ["LLM_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout_s=30.0,
        max_retries=3,
        base_delay_s=0.5,
    ) as client:
        request = LLMRequest(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Explain exponential backoff in one sentence."}],
            temperature=0.7,
            max_tokens=100,
        )

        try:
            response = await client.complete(request)
            print(f"Response: {response.content}")
            print(f"Attempts: {response.attempts}")
            print(f"Latency: {response.latency_ms:.1f} ms")
            print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
        except LLMBadRequestError as e:
            print(f"Client error (not retried): {e}")
        except LLMRateLimitError as e:
            print(f"Rate limited after retries: {e}")

asyncio.run(main())
```

### Streaming Completion with Token Usage

```python
import asyncio
import os
from llm_client import AsyncLLMClient, LLMRequest

async def stream_demo():
    async with AsyncLLMClient(api_key=os.environ["LLM_API_KEY"]) as client:
        request = LLMRequest(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
        )

        final_usage = {}
        async for token in client.stream(
            request,
            on_usage=lambda usage: final_usage.update(usage.model_dump()),
        ):
            print(token, end="", flush=True)

        print("\n\nUsage:", final_usage)

asyncio.run(stream_demo())
```

---

## API Reference & Service Endpoints

### 1. `POST /v1/complete`
Performs a non-streaming chat completion with automatic retries and timing.

**Request Body (`LLMRequest`):**
```json
{
  "model": "openai/gpt-oss-120b",
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response (`LLMResponse` - 200 OK):**
```json
{
  "content": "Hello! How can I assist you today?",
  "model": "openai/gpt-oss-120b",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 9,
    "total_tokens": 21
  },
  "latency_ms": 234.5,
  "attempts": 1
}
```

### 2. `POST /v1/stream`
Streams chat completion tokens using Server-Sent Events (SSE).

**Stream Event Flow:**
```
data: {"token": "Hello"}

data: {"token": " world"}

data: {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, "model": "openai/gpt-oss-120b"}

data: [DONE]
```

### 3. `POST /v1/simulate`
Simulates provider failures against an in-memory `httpx.MockTransport` without making external API calls. Used by the Acceptance Test Studio.

**Request:**
```json
{
  "scenario": "bad_request_400",
  "max_retries": 3,
  "base_delay_s": 0.5
}
```

**Server-Sent Event Stream:** Emits real-time `attempt` events and terminal `result` events containing timestamps, jitter delays, and policy decisions.

### 4. System Health Endpoints
- `GET /health` ➔ `{"status": "ok"}`
- `GET /config` ➔ `{"default_model": "...", "base_url": "..."}`

---

## Acceptance Test Matrix

The test suite validates compliance with the Week 1 requirements:

| # | Test Scenario | Trigger | Expected Client Action | Assertion |
|---|---|---|---|---|
| 1 | **Clean 200 OK** | Valid response on Attempt 1 | Succeeded on attempt 1 | `attempts == 1`, valid `content` |
| 2 | **Transient 500 Recovery** | 500 on att. 1-2, 200 on att. 3 | Retries twice with jitter, recovers | `attempts == 3`, valid `content` |
| 3 | **Persistent 429 Exhaustion** | 429 on all attempts | Retries up to `max_retries`, stops | Raises `LLMRateLimitError`, attempt count == `max_retries` |
| 4 | **Bad Request Fast-Fail** | 400 Bad Request | **Fails immediately on attempt 1** | Raises `LLMBadRequestError`, `attempts == 1` |
| 5 | **Connection Timeout** | Network timeout on all attempts | Retries with backoff, stops | Raises `LLMTimeoutError`, attempt count == `max_retries` |
| 6 | **Malformed Response** | 200 OK with missing required keys | **Fails immediately on attempt 1** | Raises `LLMMalformedResponseError`, `attempts == 1` |
| 7 | **Streaming Tokens** | Chunked SSE stream | Emits tokens sequentially | Yields matching token list |
| 8 | **Streaming Connect Retry** | Connection drop before handshake | Retries connect, then streams | Successfully streams after retry |

Run all acceptance tests locally:
```bash
pytest tests/ -v
```

---

## Docker Deployment

Build and run the containerized service:

```bash
# Build the production image
docker build -t llm-client .

# Run container with API key passed via environment variable
docker run -d \
  --name llm-client-service \
  -p 8000:8000 \
  -e LLM_API_KEY="your-api-key-here" \
  llm-client

# Check logs
docker logs -f llm-client-service
```

Test the running container:
```bash
curl http://localhost:8000/health
```
