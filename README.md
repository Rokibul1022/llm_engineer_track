# 🚀 LLM Engineer Track: From Async Python to Production Agents & RAG

> **An end-to-end, production-grade LLM engineering repository comprising 24 progressive technical milestones.**  
> Advancing from resilient asynchronous Python networking and robust streaming clients to deep Transformer inference benchmarking, autonomous multi-agent systems, and enterprise retrieval-augmented generation (RAG) architectures.

---

## 📑 Table of Contents
1. [Curriculum Roadmap (24 Technical Milestones)](#-curriculum-roadmap-24-technical-milestones)
2. [Milestone Deep Dives](#-milestone-deep-dives)
   - [Phase 1 Week 1: Resilient Async LLM Client & Distributed Fault Tolerance](#phase-1-week-1-resilient-async-llm-client--distributed-fault-tolerance)
   - [Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics](#phase-1-week-2-inference-pipeline-token-benchmarking--decoding-dynamics)
3. [System Architecture](#-system-architecture)
4. [Inference Pipeline: Theory to Codebase Mapping](#-inference-pipeline-theory-to-codebase-mapping)
5. [Empirical Benchmark Results (`openai/gpt-oss-120b`)](#-empirical-benchmark-results-openaigpt-oss-120b)
6. [Quickstart & Running Locally](#-quickstart--running-locally)
   - [Running the Unit Test Suite](#1-running-the-automated-test-suite)
   - [Running the Standalone Benchmark CLI](#2-running-the-benchmark-cli-runner)
   - [Running the Interactive Multipage Streamlit Studio](#3-launching-the-interactive-streamlit-studio)
7. [Repository Structure](#-repository-structure)
8. [Tradeoffs & Known Limitations](#-tradeoffs--known-limitations)

---

## 🗺️ Curriculum Roadmap (24 Technical Milestones)

| Phase | Milestone / Focus | Status | Core Deliverables & Capabilities |
| :--- | :--- | :---: | :--- |
| **Phase 1: Week 1** | **Resilient Async LLM Client & Latency Insights** | ✅ Completed | Production async client, AWS Full Jitter backoff, circuit-breaker fast-fail policy, TTFT & decode token decomposition, zero-egress fault injection simulation harness. |
| **Phase 1: Week 2** | **Inference Pipeline & Token Benchmarking** | ✅ Completed | Transformer pipeline architecture mapping, server-side TTFT & throughput benchmarking engine, temperature/top-p nondeterminism analysis, multipage Streamlit interactive studio. |
| **Phase 1: Week 3** | **Streaming Architectures & SSE Protocol** |  ✅ done | Real-time token delivery pipelines, backpressure handling, bi-directional event transport, connection lifecycle recovery. |
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

## 🏛️ System Architecture

```
                                      +------------------------------------+
                                      |          Browser / Client          |
                                      +-----------------+------------------+
                                                        |
                                                        | HTTP JSON / SSE Stream
                                                        v
+--------------------------------------------------------------------------------------------------+
| FastAPI Gateway (app.py) & Streamlit Demo Studio (streamlit_demo.py)                             |
|  - Validates request payload against Pydantic LLMRequest                                         |
|  - Enforces domain exceptions (LLMRateLimitError, LLMBadRequestError, LLMTimeoutError)           |
|  - Streams Server-Sent Events (SSE) token-by-token with monotonic telemetry                      |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Calls client.complete() / client.stream()
                                                        v
+--------------------------------------------------------------------------------------------------+
| AsyncLLMClient Engine (llm_client/client.py & llm_client/benchmark.py)                           |
|  - Decorrelated Full-Jitter Exponential Backoff                                                  |
|  - Fast-fail circuit breaker for 4xx / malformed schemas                                         |
|  - High-resolution wall-clock monotonic timing via time.perf_counter()                           |
|  - Token generation benchmarking suite across prompt lengths and temperature sweeps              |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Outbound HTTPS / TLS Handshake
                                                        v
+--------------------------------------------------------------------------------------------------+
| Upstream LLM Serving Infrastructure (openai/gpt-oss-120b on Groq Cloud)                          |
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

*(For the complete architectural walkthrough, see [`PIPELINE.md`](phase1-week1/llm-client/PIPELINE.md) and [`architecture.md`](architecture.md)).*

---

## 📊 Empirical Benchmark Results (`openai/gpt-oss-120b`)

Real empirical telemetry collected by running the automated benchmark suite (`python scripts/run_benchmark.py --repeats 3 --delay 2.1`) against `openai/gpt-oss-120b` (persisted in [`benchmark_results.json`](phase1-week1/llm-client/benchmark_results.json)):

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

> **Key Empirical Insight**: On the Long Prompt at `temperature=0.0`, 2 distinct text variations were observed across 3 repeats. This empirically verifies that in distributed GPU serving environments, floating-point reduction non-associativity across concurrent batches can introduce subtle divergences even during greedy sampling.

---

## ⚡ Quickstart & Running Locally

### Prerequisites
- Python 3.10+
- Groq Cloud API Key (or use simulated mock mode without any credentials)

```bash
# Clone and enter the repository
git clone https://github.com/Rokibul1022/llm_engineer_track.git
cd llm_engineer_track/phase1-week1/llm-client

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

Configure `.env` (copied from `.env.example`):
```ini
LLM_API_KEY=your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_DEFAULT_MODEL=openai/gpt-oss-120b
```

---

### 1. Running the Automated Test Suite
The test suite utilizes `respx` to mock the HTTP transport layer—enabling instant, hermetic execution with zero external network egress:

```powershell
pytest tests/ -v
```
**Result**: **11/11 tests passing (100%)** in ~0.45s (verifying retry policies, circuit breakers, SSE streaming, TTFT calculations, and nondeterminism detection).

---

### 2. Running the Benchmark CLI Runner

**A. Live interactive single-prompt streaming with real-time telemetry:**
```powershell
python scripts/run_benchmark.py --prompt "Explain what attention does in Transformer models in 2 sentences."
```

**B. Full Cartesian benchmark suite (3 prompts × 3 settings × 3 repeats):**
```powershell
# Live API with rate-limit pacing delay:
python scripts/run_benchmark.py --repeats 3 --delay 2.1

# Or offline with mock streaming transport:
python scripts/run_benchmark.py --mock --repeats 3
```

---

### 3. Launching the Interactive Streamlit Studio

Run the dual-module multipage Streamlit application:

```powershell
streamlit run streamlit_demo.py
```
Open **`http://localhost:8501`** in your browser:

1. **`1_Week1_Fault_Injection` (Phase 1 Week 1 Studio)**:
   - **⚡ Tab 1: Live Client, Speed Insights & Execution Pipeline**: Live token streaming, real-time 5-stage pipeline stepper, 4-row Speed Insights HUD, physical latency progress bars (Prefill % vs Decode %), and formula call stacks.
   - **🧪 Tab 2: Acceptance Test & Fault Injection Studio**: Interactive simulation of all 6 failure modes (500 recovery, 429 rate limits, timeouts, 400 fast-fail, malformed 200) with live retry logs and backoff countdowns.
2. **`2_Week2_Benchmark` (Phase 1 Week 2 Lab)**:
   - **💬 Tab 1: Live Interactive Streaming**: Test custom prompts live with instantaneous TTFT and decode throughput tracking.
   - **📊 Tab 2: Nondeterminism Benchmark Suite**: Run multi-repeat temperature sweeps (0.0, 0.7, 1.2) with side-by-side text divergence comparison cards.

---

## 📁 Repository Structure

```text
llm_engineer_track/
├── README.md                                  # Comprehensive track overview and roadmap
├── answers.md                                 # Core knowledge questions & technical answers
├── architecture.md                            # Transformer inference pipeline specification
├── architecture.png                           # Visual architecture diagram
├── phase1-week1/
│   └── llm-client/
│       ├── .env.example                       # Environment configuration template
│       ├── app.py                             # FastAPI proxy gateway & SSE endpoint
│       ├── benchmark_results.json             # Persisted empirical benchmark measurements
│       ├── test_report.json                   # Automated test report & verification metadata
│       ├── PIPELINE.md                        # End-to-end pipeline mapping to codebase
│       ├── LIMITATIONS.md                     # Engineering tradeoffs & known constraints
│       ├── ANSWERS.md                         # Milestone technical answers
│       ├── streamlit_demo.py                  # Main Streamlit application entrypoint
│       ├── llm_client/
│       │   ├── __init__.py                    # Public exports
│       │   ├── client.py                      # AsyncLLMClient with Full Jitter backoff
│       │   ├── schemas.py                     # Pydantic v2 data models (LLMRequest, Usage)
│       │   ├── exceptions.py                  # Typed exception hierarchy
│       │   └── benchmark.py                   # Benchmark engine & nondeterminism analyzer
│       ├── pages/
│       │   ├── 1_Week1_Fault_Injection.py     # Week 1 Live Client & Fault Injection Studio
│       │   └── 2_Week2_Benchmark.py           # Week 2 Token Benchmark & Live Generation Lab
│       ├── scripts/
│       │   └── run_benchmark.py               # CLI benchmark runner with live streaming
│       ├── static/
│       │   └── index.html                     # Visual demonstration dashboard (FastAPI)
│       └── tests/
│           ├── test_client.py                 # Resilient client & retry policy tests (8 tests)
│           └── test_benchmark.py              # Benchmark & nondeterminism unit tests (3 tests)
```

---

## ⚖️ Tradeoffs & Known Limitations

1. **Client-Side API Boundary vs. Model Internal Weights**:
   The client interacts across an external HTTP/SSE API boundary. Token counts and completion latencies reflect provider-reported usage rather than directly instrumented GPU kernel timers.
2. **TTFT Latency Composition**:
   Client-measured TTFT includes client serialization, WAN internet transit, provider API gateway routing, and GPU queueing alongside the true model prefill time.
3. **Provider Rate Limits (RPM/TPM)**:
   Free-tier cloud provider quotas (e.g. Groq 30 RPM) require pacing delays (`--delay 2.1`) during large Cartesian benchmark sweeps to prevent transient HTTP 429 throttling.
4. **Streamlit Synchronous Worker Loop**:
   Executing asynchronous benchmark loops inside Streamlit via `asyncio.run()` blocks the session thread during bulk multi-repeat executions.

*(For detailed architectural tradeoffs, see [`LIMITATIONS.md`](phase1-week1/llm-client/LIMITATIONS.md)).*

---

## 🤝 Contribution & Integration

- **Active Development Branch**: [`phase1-week2`](https://github.com/Rokibul1022/llm_engineer_track/tree/phase1-week2)
- **Base Integration Branch**: `phase1-week1`
- **Pull Request**: [Open Pull Request on GitHub](https://github.com/Rokibul1022/llm_engineer_track/pull/new/phase1-week2)
- **Author**: Rokibul
