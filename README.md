# 🚀 LLM Engineer Track: From Async Python to Production Agents & RAG

> **An end-to-end, production-grade LLM engineering repository comprising 24 progressive technical milestones.**  
> Advancing from resilient asynchronous Python networking and robust streaming clients to deep Transformer inference benchmarking, schema-validated structured extraction pipelines, autonomous multi-agent systems, and enterprise retrieval-augmented generation (RAG) architectures.

---

## 📑 Table of Contents
1. [Curriculum Roadmap (24 Technical Milestones)](#-curriculum-roadmap-24-technical-milestones)
2. [Milestone Deep Dives](#-milestone-deep-dives)
   - [Phase 1 Week 1: Resilient Async LLM Client & Distributed Fault Tolerance](#phase-1-week-1-resilient-async-llm-client--distributed-fault-tolerance)
   - [Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics](#phase-1-week-2-inference-pipeline-token-benchmarking--decoding-dynamics)
   - [Phase 1 Week 3: Schema-Validated Extraction API Service & Pipeline Auditor Studio](#phase-1-week-3-schema-validated-extraction-api-service--pipeline-auditor-studio)
3. [System Architecture](#-system-architecture)
4. [Inference Pipeline: Theory to Codebase Mapping](#-inference-pipeline-theory-to-codebase-mapping)
5. [Empirical Benchmark Results (`openai/gpt-oss-120b`)](#-empirical-benchmark-results-openaigpt-oss-120b)
6. [Quickstart & Running Locally](#-quickstart--running-locally)
   - [Running the Unit Test Suite (Week 1)](#1-running-the-automated-unit-test-suite)
   - [Running the Standalone Benchmark CLI (Week 2)](#2-running-the-benchmark-cli-runner)
   - [Running the FastAPI Extraction Backend & HTML Auditor (Week 3)](#3-running-the-fastapi-extraction-service--auditor-web-ui)
   - [Running the Interactive Unified Streamlit Studio](#4-launching-the-interactive-streamlit-studio)
7. [Repository Structure](#-repository-structure)
8. [Tradeoffs & Known Limitations](#-tradeoffs--known-limitations)

---

## 🗺️ Curriculum Roadmap (24 Technical Milestones)

| Phase | Milestone / Focus | Status | Core Deliverables & Capabilities |
| :--- | :--- | :---: | :--- |
| **Phase 1: Week 1** | **Resilient Async LLM Client & Latency Insights** | ✅ Completed | Production async client, AWS Full Jitter backoff, circuit-breaker fast-fail policy, TTFT & decode token decomposition, zero-egress fault injection simulation harness. |
| **Phase 1: Week 2** | **Inference Pipeline & Token Benchmarking** | ✅ Completed | Transformer pipeline architecture mapping, server-side TTFT & throughput benchmarking engine, temperature/top-p nondeterminism analysis, multipage Streamlit interactive studio. |
| **Phase 1: Week 3** | **Schema-Validated Extraction API & Pipeline Auditor** | ✅ Completed | Pydantic v2 data models, Groq function calling on `openai/gpt-oss-120b`, bounded self-correction retry loop, non-LLM semantic sanity checks, FastAPI `/extract` and SSE `/extract/stream`, unified Streamlit studio integration. |
| **Phase 1: Week 4** | **Observability, Distributed Tracing & Token Economics** | 📋 Scheduled | OpenTelemetry integration, distributed trace context propagation, granular per-token cost ledger, latency SLA monitoring. |
| **Phase 2: Weeks 5–10**| **Production Retrieval-Augmented Generation (RAG)** | 📋 Scheduled | Hybrid search (dense vectors + BM25 sparse), Reciprocal Rank Fusion (RRF), Cross-Encoder dynamic reranking, chunking strategies, contextual compression. |
| **Phase 3: Weeks 11–18**| **Autonomous Multi-Agent Systems & Tool Calling** | 📋 Scheduled | ReAct / Plan-and-Solve engines, structured function calling, sandbox code execution, human-in-the-loop validation checkpoints, multi-agent orchestration. |
| **Phase 4: Weeks 19–24**| **Enterprise Evaluation, Guardrails & Cloud Deployment** | 📋 Scheduled | Automated red teaming, LLM-as-a-judge evaluation frameworks, NeMo / Llama Guard safety guardrails, Kubernetes deployment, circuit breakers at enterprise scale. |

---

## 🔬 Milestone Deep Dives

### Phase 1 Week 1: Resilient Async LLM Client & Distributed Fault Tolerance

The foundation of any production LLM platform is an asynchronous client that handles the harsh reality of distributed LLM serving: persistent rate limits, transient provider crashes, socket timeouts, and malformed outputs.

- **Non-Blocking Asynchronous Core**: Built on `httpx.AsyncClient` with typed contracts (`LLMRequest`, `LLMResponse`, `Usage`) enforced via Pydantic v2.
- **AWS Full Jitter Backoff Algorithm**: Prevents thundering herd retry storms when distributed nodes hit rate limits simultaneously:
  $$\text{Sleep Delay} = \text{Uniform}\left(0, \, \min\left(\text{MaxDelay}, \, \text{BaseDelay} \times 2^{\text{attempt}}\right)\right)$$
- **Circuit-Breaker Fast-Fail Policy**:
  - `429 Rate Limit` & `5xx Server Error`: **Retryable** with exponential backoff and jitter.
  - `4xx Client Error` & `200 Malformed JSON`: **Non-Retryable**; fast-fails immediately on Attempt 1 with zero retry quota waste.
- **Streaming Handshake Safety**: Retries only apply while establishing the initial HTTP connection. Once tokens begin flowing, mid-stream disconnects are surfaced immediately to prevent token duplication to users.

---

### Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics

Week 2 connects theoretical Transformer mechanics to empirical measurements, dissecting the physical latency profile of Large Language Model generation.

- **The Two Computational Regimes**:
  1. **Prefill Phase (Prompt Ingestion)**: Ingests all prompt tokens concurrently via matrix multiplications ($QK^T / \sqrt{d_k}$) to populate the Key-Value (KV) cache. Measured client-side as **Time to First Token (TTFT)**. Scales quadratically ($O(N^2)$) with prompt length.
  2. **Decode Phase (Autoregressive Token Generation)**: Iteratively produces one token per forward pass by reading cached KV states from High Bandwidth Memory (HBM). Measured as **Decode Throughput (Tokens/sec)**.
- **Decoding Controls & Output Nondeterminism**:
  - **Temperature ($T$)**: Reshapes the softmax probability distribution over vocabulary logits ($P(x_i) = \frac{\exp(z_i/T)}{\sum \exp(z_j/T)}$). Low temperature ($T \to 0$) collapses toward greedy argmax selection; higher values ($T \ge 0.7$) broaden lexical variation.
  - **Top-P (Nucleus Sampling)**: Truncates candidates to the minimal probability subset whose cumulative sum crosses threshold $p$, allowing the candidate pool to dynamically expand or contract based on model confidence.
  - **Hardware Floating-Point Nondeterminism**: Demonstrates empirically why modern multi-GPU tensor-parallel inference engines can produce slight output variations even at $T=0.0$ due to non-associative IEEE 754 floating-point reduction across distributed kernels.

---

### Phase 1 Week 3: Schema-Validated Extraction API Service & Pipeline Auditor Studio

Week 3 shifts from raw text generation to production-grade, schema-validated structured data extraction using Groq's high-speed function calling API (`openai/gpt-oss-120b`).

- **Typed Pydantic Contracts (`schemas.py`)**:
  - `TicketExtraction`: Enforces typed `intent`, `urgency`, typed `entities` list, `summary` ($\le 200$ chars), and numeric `confidence`.
  - `InvoiceExtraction`: Enforces strict float amount parsing, ISO `due_date`, currency enums, and vendor metadata.
- **Bounded Self-Correction Retry Loop (`extractor.py`)**:
  - Intercepts Pydantic `ValidationError` and JSON decoding failures.
  - Feeds the exact validation error back to the assistant in a `tool` role message (budget of 2 retries), allowing the LLM to self-correct inline.
- **Groq Proxy 400 Interception (`groq_client.py`)**:
  - Transparently intercepts Groq's `tool_use_failed` 400 proxy responses, unpacks `failed_generation`, and feeds it into the validator/retry pipeline.
- **Semantic Sanity Checks (Non-LLM Heuristic Guard)**:
  - Catches the failure class schema validation cannot see: outputs that are schema-valid but factually or logically wrong (e.g. sarcasm detection, negation inversion, ungrounded summaries, negative balances, missing due dates).
- **FastAPI Endpoints & SSE Streaming (`main.py`)**:
  - `POST /extract`: Synchronous extraction returning structured JSON, attempt counts, and sanity warning flags.
  - `POST /extract/stream`: Server-Sent Events (SSE) streaming partial argument deltas with single-pass final buffer validation.
- **Pipeline Auditor Studio**:
  - Integrated into the Streamlit studio as **Tab 1: API Extraction** with a 5-stage backend stepper, metric HUD, and attempt-by-attempt self-correction recovery cards.
  - Standalone web auditor served at `http://localhost:8000/`.

*(For the complete dedicated Week 3 documentation, see [`phase1-week3/README.md`](phase1-week3/README.md)).*

---

## 🏛️ System Architecture

```
                                      +------------------------------------+
                                      |          Browser / Client          |
                                      +-----------------+------------------+
                                                        |
                                                        | HTTP JSON / SSE Stream
                                                        v
+--------------------------------------------------------------------------------------------------+
| FastAPI Gateway (app.py & main.py) & Streamlit Demo Studio (streamlit_demo.py)                   |
|  - Validates request payload against Pydantic LLMRequest & Extraction schemas                    |
|  - Enforces domain exceptions (LLMRateLimitError, ExtractionFailure)                             |
|  - Streams Server-Sent Events (SSE) token-by-token with monotonic telemetry                      |
|  - Interactive 5-stage backend execution pipeline stepper & self-correction audit cards          |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Calls client.complete() / extract()
                                                        v
+--------------------------------------------------------------------------------------------------+
| Core Execution Engine (llm_client/client.py & phase1-week3/extractor.py)                         |
|  - Decorrelated Full-Jitter Exponential Backoff                                                  |
|  - Bounded Self-Correction Retry Loop (feeds ValidationError back to LLM)                        |
|  - Non-LLM Semantic Sanity Guard (negation, sarcasm, balance checks)                             |
|  - High-resolution wall-clock monotonic timing via time.perf_counter()                           |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Outbound HTTPS / TLS Handshake
                                                        v
+--------------------------------------------------------------------------------------------------+
| Upstream LLM Serving Infrastructure (openai/gpt-oss-120b on Groq Cloud)                          |
|  - Function calling / Tool choice enforcement                                                    |
|  - Prefill Phase: Subword Tokenization -> Embeddings -> Transformer Attention -> KV Cache Init   |
|  - Decode Phase: Logits -> Softmax Temperature / Top-P Sampling -> Autoregressive Streaming     |
+--------------------------------------------------------------------------------------------------+
```

---

## 🔄 Inference Pipeline: Theory to Codebase Mapping

| Pipeline Stage | Theoretical Mechanism | Concrete Codebase Mapping |
| :--- | :--- | :--- |
| **1. Tokenization** | Subword BPE discretization mapping strings to vocabulary IDs ($V \approx 128\text{k}$). | In [`schemas.py`](phase1-week1/llm-client/llm_client/schemas.py), `Usage` schema records `prompt_tokens` emitted by provider SSE frames. |
| **2. Embeddings** | Continuous vector projection ($W_E \in \mathbb{R}^{|V| \times d_{\text{model}}}$) + RoPE positional rotations. | Model selection via `LLMRequest.model` (defaulting to `openai/gpt-oss-120b`). |
| **3. Attention (Prefill)** | Multi-head self-attention ($QK^T / \sqrt{d_k}$) across prompt tokens. Attention cost scales with $O(N^2)$. | Benchmarked in [`benchmark.py`](phase1-week1/llm-client/llm_client/benchmark.py); TTFT scaling measured across short, medium, and long prompts. |
| **4. Logits & Sampling** | Softmax projection temperature scaling and Top-P nucleus probability cutoff. | Configured via `LLMRequest.temperature` and `LLMRequest.top_p`; evaluated in `summarize()` across repeated runs. |
| **5. Autoregressive Decode** | One forward pass per token; past key/value states preserved in high-speed GPU KV Cache. | Directly driven by `async for chunk in client.stream()` in [`client.py`](phase1-week1/llm-client/llm_client/client.py) and SSE loops in [`app.py`](phase1-week1/llm-client/app.py). |
| **6. Detokenization** | Integer token IDs decoded back into UTF-8 characters and emitted to client. | Read from SSE `delta.content` chunks and rendered live in the CLI and Streamlit interfaces. |

---

## 📊 Empirical Benchmark Results (`openai/gpt-oss-120b`)

Real empirical telemetry collected by running the automated benchmark suite against `openai/gpt-oss-120b` (persisted in [`benchmark_results.json`](phase1-week1/llm-client/benchmark_results.json)):

```text
================================================================================================================
 [BENCHMARK] INFERENCE BENCHMARK SUMMARY TABLE
================================================================================================================
Prompt Size    | Temp   | Top-P   | Runs   | Prompt Tok   | Comp Tok   | TTFT (ms)   | Tokens/s   | Identical?  
---------------+--------+---------+--------+--------------+------------+-------------+------------+-------------
Short (36c)    | 0.0    | 1.0     | 3      | 81           | 120        | 605.2       | 199.2      | YES (det)   
Short (36c)    | 0.7    | 0.9     | 3      | 81           | 115        | 592.4       | 187.5      | NO (3 var)  
Short (36c)    | 1.2    | 1.0     | 3      | 81           | 123        | 695.5       | 172.3      | NO (3 var)  
Medium (162c)  | 0.0    | 1.0     | 3      | 99           | 128        | 546.0       | 179.3      | YES (det)   
Medium (162c)  | 0.7    | 0.9     | 3      | 99           | 128        | 627.6       | 169.3      | NO (3 var)  
Medium (162c)  | 1.2    | 1.0     | 3      | 99           | 128        | 622.8       | 189.3      | NO (3 var)  
Long (758c)    | 0.0    | 1.0     | 3      | 214          | 128        | 698.7       | 158.7      | NO (2 var)* 
Long (758c)    | 0.7    | 0.9     | 3      | 214          | 128        | 661.1       | 164.4      | NO (3 var)  
Long (758c)    | 1.2    | 1.0     | 3      | 214          | 128        | 651.4       | 170.3      | NO (3 var)  
---------------+--------+---------+--------+--------------+------------+-------------+------------+-------------
```

---

## ⚡ Quickstart & Running Locally

### Prerequisites
- Python 3.10+
- Groq Cloud API Key configured in `.env`:
  ```ini
  GROQ_API_KEY=gsk_your_groq_api_key_here
  LLM_DEFAULT_MODEL=openai/gpt-oss-120b
  ```

---

### 1. Running the Automated Unit Test Suite
```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\pytest.exe" .\phase1-week1\llm-client\tests\ -v
```

---

### 2. Running the Benchmark CLI Runner
```powershell
python phase1-week1/llm-client/scripts/run_benchmark.py --prompt "Explain attention mechanisms in 2 sentences."
```

---

### 3. Running the FastAPI Extraction Service & Auditor Web UI (Week 3)
```powershell
cd phase1-week3
& "..\phase1-week1\llm-client\.venv\Scripts\uvicorn.exe" main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser to inspect live pipeline execution and Server-Sent Events parameter deltas.

---

### 4. Launching the Interactive Streamlit Studio
```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\streamlit.exe" run ".\phase1-week1\llm-client\streamlit_demo.py"
```
Open **`http://localhost:8501`** in your browser to access:
- **🛡️ Tab 1: API Extraction — Pipeline Auditor View (Week 3)**: Test structured extraction, adversarial samples, and watch self-correcting retry recovery live.
- **⚡ Tab 2: Live Client, Speed Insights & Execution Pipeline (Week 1 & 2)**: Live token streaming with TTFT vs. decode throughput decomposition.
- **🧪 Tab 3: Acceptance Test & Fault Injection Studio (Week 1)**: Interactive simulations of 429 rate limits, 5xx server crashes, malformed outputs, and fast-fail circuit breakers.

---

## 📁 Repository Structure

```text
llm_engineer_track/
├── README.md                                  # Unified master roadmap & curriculum guide
├── prompt.md                                  # Curated test prompts & adversarial error cases
├── phase1-week1/
│   └── llm-client/
│       ├── streamlit_demo.py                  # Main unified Streamlit studio entrypoint
│       ├── llm_client/
│       │   ├── client.py                      # AsyncLLMClient with Full Jitter backoff
│       │   ├── schemas.py                     # Pydantic v2 data models
│       │   ├── exceptions.py                  # Domain exception hierarchy
│       │   └── benchmark.py                   # Benchmark & nondeterminism engine
│       ├── pages/
│       │   ├── 1_Week1_Fault_Injection.py     # Fault Injection Studio
│       │   ├── 2_Week2_Benchmark.py           # Token Benchmark Lab
│       │   └── 3_API_Extraction.py           # Week 3 Extraction Studio
│       └── tests/                             # Unit test suite
├── phase1-week3/
│   ├── README.md                              # Dedicated Week 3 documentation
│   ├── schemas.py                             # Pydantic extraction models (Ticket & Invoice)
│   ├── prompts.py                             # System prompts & dynamic tool schema builder
│   ├── groq_client.py                         # Groq HTTP client & 400 proxy error interceptor
│   ├── extractor.py                           # Self-correcting retry loop & semantic sanity guard
│   ├── main.py                                # FastAPI app (/extract, /extract/stream)
│   ├── static/index.html                      # Standalone Pipeline Auditor web UI
│   └── week3-test-report.md                   # 100% verified empirical test report
```

---

## ⚖️ Tradeoffs & Known Limitations

1. **Client-Side API Boundary vs. Internal Hardware Timers**: Token counts and latencies reflect provider-reported usage across WAN HTTP/SSE connections rather than direct GPU kernel telemetry.
2. **Grammar Enforcement vs. LLM Freedom**: Strict tool-calling constraints reduce hallucinations but require fallback heuristics when inputs completely lack required target fields.
3. **Bounded Retries**: Self-correction is capped at 2 retries to prevent unbounded token expenditure on unrecoverable inputs.
