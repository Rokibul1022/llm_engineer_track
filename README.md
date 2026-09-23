# 🚀 LLM Engineer Track: From Async Python to Production Agents & RAG

> **An end-to-end, production-grade LLM engineering curriculum comprising 24 progressive technical milestones across 6 core phases.**  
> Advancing from resilient asynchronous Python networking and robust streaming clients to deep Transformer inference benchmarking, schema-validated structured extraction, million-document RAG architectures, stateful multi-agent workflows, and GPU inference optimization.

---

## 📑 Table of Contents
1. [Curriculum Philosophy & Operating Rhythm](#-curriculum-philosophy--operating-rhythm)
2. [Master Curriculum Roadmap (24 Technical Milestones)](#-master-curriculum-roadmap-24-technical-milestones)
3. [The 4 Major Real-World Client Assignments](#-the-4-major-real-world-client-assignments)
4. [The 9 Production Engineering Disciplines](#-the-9-production-engineering-disciplines)
5. [Standard Production Stack](#-standard-production-stack)
6. [Capacity Planning & Million-Document RAG Architecture](#-capacity-planning--million-document-rag-architecture)
7. [Milestone Deep Dives (Phase 1)](#-milestone-deep-dives-phase-1)
   - [Phase 1 Week 1: Resilient Async LLM Client & Latency Insights](#phase-1-week-1-resilient-async-llm-client--latency-insights-llm-client)
   - [Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics](#phase-1-week-2-inference-pipeline-token-benchmarking--decoding-dynamics-transformer)
   - [Phase 1 Week 3: Schema-Validated Structured Extraction API Service](#phase-1-week-3-schema-validated-structured-extraction-api-service-robust-api-extractor)
8. [System Architecture](#-system-architecture)
9. [Inference Pipeline: Theory to Codebase Mapping](#-inference-pipeline-theory-to-codebase-mapping)
10. [Empirical Benchmark Results (`openai/gpt-oss-120b`)](#-empirical-benchmark-results-openaigpt-oss-120b)
11. [Weekly Review Protocol & Submission Standards](#-weekly-review-protocol--submission-standards)
12. [Readiness Scorecard & Qualification Standard](#-readiness-scorecard--qualification-standard)
13. [Quickstart & Running Locally](#-quickstart--running-locally)
14. [Repository Structure](#-repository-structure)
15. [Tradeoffs & Known Limitations](#-tradeoffs--known-limitations)
16. [Branch Navigation & Integration](#-branch-navigation--integration)

---

## 🧭 Curriculum Philosophy & Operating Rhythm

This repository embodies an intensive 24-week technical path designed to advance developers from consuming LLM APIs to architecting, evaluating, optimizing, and defending high-throughput enterprise AI systems.

### 1. The 1:3 Theory-to-Practice Ratio (18 Hours / Week)
Most time is spent writing production code, building test suites, and profiling systems—not passively consuming tutorials.
- **4 Hours**: Academic papers, free lectures, and specifications.
- **6 Hours**: Hands-on technical coding labs and failure injection experiments.
- **6 Hours**: Real-world project implementation and architectural integration.
- **2 Hours**: Rigorous code review, benchmark validation, and technical discussion.

### 2. Concrete Production Rules
- **No Notebook-Only Submissions**: Every milestone requires a clean Git branch, pull request, typed code contracts, automated tests, reproducible benchmark scripts, architecture documentation, and live runnable demos.
- **Abstractions are Tools, Not Foundations**: Developers must master native Python async, raw HTTP protocols, vector index mechanics, and state graph transitions before using frameworks like LangChain or LangGraph. Framework abstractions never replace understanding.

---

## 🗺️ Master Curriculum Roadmap (24 Technical Milestones)

The 24 milestones are organized into 6 distinct phases. Each phase culminates in a deployable system deliverable and a rigorous technical review.

| Phase & Deliverable | Wk | Milestone Name | Status | Topics & Core Deliverables |
| :--- | :---: | :--- | :---: | :--- |
| **Phase 1: LLM Foundations**<br>*(Weeks 1–3)*<br><br>🎯 **Deliverable:**<br>Production LLM API Service | **01** | **Python Engineering, Async & Tokenization** | ✅ Completed<br>(`llm-client`) | Async I/O, `httpx`, Pydantic v2 schemas, AWS Full Jitter backoff, circuit-breaker fast-fail, TTFT/decode latency instrumentation, zero-egress fault simulation. |
| | **02** | **Transformers, Attention & Decoding Dynamics** | ✅ Completed<br>(`transformer`) | Prefill vs. decode profiling, KV cache dynamics, TTFT decomposition, temperature/top-p sampling, multi-GPU float32 nondeterminism analysis, Streamlit benchmarking lab. |
| | **03** | **Prompts, Tool Calling & Structured Outputs** | ✅ Completed<br>(`robust-api-extractor`) | Function calling (`openai/gpt-oss-120b`), bounded self-correcting retry loop with schema error feedback, semantic sanity guards, SSE streaming, Pipeline Auditor Studio. |
| **Phase 2: RAG Engineering**<br>*(Weeks 4–7)*<br><br>🎯 **Deliverable:**<br>Evaluated Document Q&A System | **04** | **Document Ingestion & Chunking** | 📋 Scheduled | PDF/HTML/DOCX parsing, OCR integration, text normalization, metadata extraction, structural/semantic chunking, source lineage tracking. |
| | **05** | **Embeddings & Vector Search** | 📋 Scheduled | Continuous representations, cosine similarity, vector databases, HNSW/IVF indexing, recall vs. latency tradeoffs on labeled datasets. |
| | **06** | **Hybrid Retrieval & Reranking** | 📋 Scheduled | BM25 sparse lexical search, dense semantic search, Reciprocal Rank Fusion (RRF), Cross-Encoder dynamic reranking, query rewriting. |
| | **07** | **Answer Generation & RAG Evaluation** | 📋 Scheduled | Dynamic context assembly, citation mapping, hallucination detection, Golden dataset evaluation (Recall@k, MRR, nDCG, faithfulness). |
| **Phase 3: Production RAG at Scale**<br>*(Weeks 8–11)*<br><br>🎯 **Deliverable:**<br>ADR, Benchmark Report & Scaled API | **08** | **Restartable Ingestion at 100k Documents** | 📋 Scheduled | Capacity sizing (tokens, chunks, raw vectors), distributed queue orchestration, idempotent restartable ingestion, backpressure control. |
| | **09** | **Scaling to 1 Million Documents** | 📋 Scheduled | 1M document benchmark, index footprint, ingestion throughput, dense vs. hybrid recall, nDCG@k, p95/p99 query latency under load. |
| | **10** | **Updates, Deletion, ACLs & Multi-Tenancy** | 📋 Scheduled | Incremental document updates, soft/hard deletion, versioning, document-level ACL permission filters, multi-tenant data isolation. |
| | **11** | **Concurrent Load & Bottleneck Diagnosis** | 📋 Scheduled | High-concurrency stress testing, p50/p95/p99 latency decomposition, bottleneck profiling (I/O, database locks, GPU saturation). |
| **Phase 4: LangChain & LangGraph**<br>*(Weeks 12–15)*<br><br>🎯 **Deliverable:**<br>Stateful Enterprise AI Workflow | **12** | **Refactoring RAG with LangChain** | 📋 Scheduled | LangChain core models, custom retrievers, tools, structured output wrappers, enterprise middleware integration. |
| | **13** | **LangGraph State & Routing** | 📋 Scheduled | State schemas, reducers, directed acyclic & cyclic graph execution, conditional routing, isolated subgraphs. |
| | **14** | **Persistence, Interrupts & Approvals** | 📋 Scheduled | Durable checkpointing, state serialization, resumable execution, human-in-the-loop approval gates, side-effect isolation. |
| | **15** | **Enterprise Agents & Observability** | 📋 Scheduled | Multi-step autonomous agent, tool error recovery, OpenTelemetry distributed tracing, full execution auditability. |
| **Phase 5: LLM Optimization**<br>*(Weeks 16–19)*<br><br>🎯 **Deliverable:**<br>Performance-Tuning Report with Reproducible Metrics | **16** | **Inference Memory & The KV Cache** | 📋 Scheduled | Transformer memory profiling, KV cache allocation, prefill vs. decode memory saturation, local model bottleneck isolation. |
| | **17** | **Quantization & Precision Tradeoffs** | 📋 Scheduled | Post-training quantization (AWQ, GPTQ, GGUF), FP16 vs. BF16 vs. FP8 vs. INT4, perplexity/quality vs. memory/throughput benchmarks. |
| | **18** | **Serving with vLLM under Concurrency** | 📋 Scheduled | vLLM production deployment, Continuous Batching, PagedAttention, multi-request concurrency stress tests. |
| | **19** | **Speculative Decoding & Tuning Report** | 📋 Scheduled | Speculative decoding with draft models, serving economics, GPU kernel profiling, comprehensive performance-tuning report. |
| **Phase 6: Production Capstone**<br>*(Weeks 20–24)*<br><br>🎯 **Deliverable:**<br>Enterprise AI Knowledge Platform + Defense | **20** | **Architecture, Threat Model & Contracts** | 📋 Scheduled | System architecture design, threat modeling (prompt injection, data exfiltration), ADRs, service contracts, data models. |
| | **21** | **Ingestion & Retrieval Integrated** | 📋 Scheduled | Scaled ingestion pipeline connected to hybrid retrieval service with live document lifecycle synchronization. |
| | **22** | **Agent, Permissions & Recovery** | 📋 Scheduled | LangGraph enterprise agent with permission enforcement, state checkpoints, and transparent disaster recovery. |
| | **23** | **Tuning, Load & Security Testing** | 📋 Scheduled | End-to-end load testing, red-teaming, automated guardrails (Llama Guard / NeMo), latency SLA enforcement. |
| **Capstone Defense** | **24** | **Production Deployment & Technical Defense** | 📋 Scheduled | Production container deployment, monitoring dashboards, live technical defense, followed by the **Unfamiliar-System Final Exam**. |

---

## 💼 The 4 Major Real-World Client Assignments

The curriculum simulates real-world enterprise requirements across four comprehensive project assignments:

### 📑 Assignment A: Enterprise Document Intelligence *(Weeks 4–11)*
- **Objective**: Build a high-performance document Q&A engine handling heterogeneous inputs (PDF, DOCX, scanned scans via OCR, HTML tables).
- **Core Requirements**: Stable document IDs, parent-child chunking, dense + BM25 hybrid search, Cross-Encoder reranking, metadata and permission-aware filtering (ACLs), incremental updates, and automated golden dataset evaluation.
- **Scale Target**: Validated on **1,000,000 documents** with documented p95 latency (< 250ms) and Recall@k benchmarks.

### 🤖 Assignment B: Enterprise Workflow Agent *(Weeks 12–15)*
- **Objective**: Build a resilient, stateful research agent capable of navigating complex enterprise knowledge bases.
- **Core Requirements**:
  1. Determine dynamically whether a user query requires RAG, database SQL lookups, or external tools.
  2. Enforce permission boundaries before querying data.
  3. Emit structured answers with exact source citations.
  4. Pause execution and seek explicit human approval before executing irreversible or consequential actions.
  5. Resume seamlessly from durable state checkpoints without re-executing completed side effects.

### ⚡ Assignment C: LLM Inference Optimization *(Weeks 16–19)*
- **Objective**: Optimize an open-weight model (e.g., Llama 3 / Mistral) for high-throughput serving on fixed GPU hardware.
- **Core Requirements**: Benchmark baseline FP16 vs. quantized INT4/AWQ/GPTQ/GGUF models. Profile compute vs. memory-bandwidth bottlenecks, measure Continuous Batching speedups in vLLM, and evaluate speculative decoding latency gains. Deliver an actionable, reproducible performance-tuning report.

### 🛡️ Assignment D: Full Production Capstone *(Weeks 20–24)*
- **Objective**: Design, build, deploy, stress-test, and defend the **Enterprise AI Knowledge Platform**.
- **The 4 System Pillars**:
  1. **Document Infrastructure**: Ingestion workers, message queues, object storage, metadata database, search indexes.
  2. **RAG Retrieval Service**: Hybrid retrieval, reranking, tenant isolation, evidence citations, and evaluation suite.
  3. **Agent Orchestration**: LangGraph workflows, controlled tool execution, checkpoints, human-in-the-loop approvals.
  4. **Inference & Operations**: Model serving, continuous batching, caching, structured logs, distributed traces, and CI/CD.

---

## 🧱 The 9 Production Engineering Disciplines

A working demo is not a maintainable production service. Every milestone reinforces nine foundational engineering disciplines:

| Discipline | Core Technical Competencies Enforced in Code |
| :--- | :--- |
| **1. Python Engineering** | Non-blocking Async I/O (`asyncio`, `httpx`), strict type annotations, Pydantic v2 schemas, hermetic `pytest` suites, memory/CPU profiling. |
| **2. Backend Services** | High-performance FastAPI services, RESTful API contracts, Server-Sent Events (SSE) streaming, authentication, structured error codes. |
| **3. Databases & Storage** | PostgreSQL with `pgvector`, database migrations, ACID transactions, B-Tree and HNSW indexing, connection pooling. |
| **4. Infrastructure** | Containerization (Docker), Linux system administration, CI/CD automated pipelines, blue/green deployments, automated rollbacks. |
| **5. Distributed Systems** | Message queues (Redis/RabbitMQ), worker pools, exponential backoff with jitter, idempotent consumers, backpressure handling. |
| **6. Observability** | OpenTelemetry distributed tracing, structured JSON logging, Prometheus metrics (p50/p95/p99 latencies, TTFT, throughput), alerting rules. |
| **7. Security & Isolation** | Multi-tenant data segregation, document-level ACL permission filtering, secret rotation, prompt injection defenses, egress boundaries. |
| **8. Reliability & Fault Tolerance** | Configurable timeouts, fast-fail circuit breakers, graceful degradation, health-check probes, dead-letter queues. |
| **9. Rigorous Evaluation** | Curated golden evaluation datasets, statistical regression suites, automated retrieval metrics (Recall@k, MRR, nDCG), LLM-as-a-judge. |

---

## 🛠️ Standard Production Stack

| Architectural Layer | Recommended Production Stack (Python-First) |
| :--- | :--- |
| **Application Layer** | Python 3.11+, FastAPI, Pydantic v2, `pytest`, `httpx` |
| **LLM Orchestration** | Native Python & HTTP first; LangChain & LangGraph for stateful graphs and checkpointing |
| **RAG & Storage** | PostgreSQL + `pgvector`, BM25 lexical engine, Cross-Encoder rerankers (`sentence-transformers`) |
| **Model Serving** | vLLM (Continuous Batching, PagedAttention), Hugging Face Transformers, Groq Cloud / Ollama |
| **Optimization & Profiling** | PyTorch, CUDA profiling tools, AutoAWQ, GPTQ, bitsandbytes |
| **Infrastructure & Queues**| Docker, Linux, Redis (caching & queues), GitHub Actions CI/CD |
| **Evaluation & Tracing** | Custom golden datasets, RAGAS / Lighteval, OpenTelemetry, Arize Phoenix / LangSmith |

---

## 📐 Capacity Planning & Million-Document RAG Architecture

### 1. Sizing Calculator: Sizing Before Provisioning
Engineers must calculate storage, memory, and compute footprints on paper before provisioning cloud infrastructure:

$$\text{Total Source Tokens} = \text{Documents} \times \text{Average Tokens per Document}$$
$$\text{Total Chunks} = \text{Documents} \times \left\lceil \frac{\text{Average Tokens}}{\text{Chunk Size}} \right\rceil$$
$$\text{Raw Vector Memory (Bytes)} = \text{Total Chunks} \times \text{Dimensions} \times \text{Bytes Per Value}$$

**Standard Million-Document Example**:
- **Documents**: 1,000,000 documents
- **Avg Tokens per Document**: 2,000 tokens $\to$ **2,000,000,000 total source tokens**
- **Chunk Size**: 500 tokens (no overlap) $\to$ **4 chunks per document**
- **Total Vectors**: **4,000,000 embedding vectors**
- **Vector Dimensions**: 1,024 dimensions (e.g. `bge-large` / `text-embedding-3-large`)
- **Precision**: 32-bit float (`4 bytes`)
- **Raw Vector Footprint**: $4,000,000 \times 1,024 \times 4 \text{ bytes} \approx \mathbf{16.384\text{ GB}}$ *(raw values only, excluding HNSW graph index overhead, metadata, and database replication buffers)*.

### 2. End-to-End Scaled Ingestion & Serving Pipeline Flow

```
[ Heterogeneous Document Sources: PDF, DOCX, HTML, Scans, Databases ]
                                |
                                v
[ Orchestrated Ingestion Workers: Parsing, Normalization, OCR, Metadata Extraction ]
                                |
                                v
[ Chunking & Enrichment Engine: Structural Chunking, Stable IDs, ACL Tags ]
                                |
                                v
[ Embedding Service: Batched Inference -> Dense Representations ]
                                |
                +---------------+---------------+
                |                               |
                v                               v
    [ Vector Index: HNSW / pgvector ]    [ Lexical Index: BM25 / Sparse ]
                |                               |
                +---------------+---------------+
                                |
                                v
[ Query Gateway: Authentication -> Tenant ACL Filter -> Parallel Retrieval ]
                                |
                                v
[ Fusion & Dynamic Reranking: Reciprocal Rank Fusion (RRF) -> Cross-Encoder ]
                                |
                                v
[ Generation & Evaluation: Context Construction -> LLM -> Citations -> Traces ]
```

---

## 🔬 Milestone Deep Dives (Phase 1)

### Phase 1 Week 1: Resilient Async LLM Client & Latency Insights (`llm-client`)

The foundation of any production LLM platform is an asynchronous client that handles the harsh reality of distributed LLM serving: persistent rate limits, transient provider crashes, socket timeouts, and malformed outputs.

- **Non-Blocking Asynchronous Core**: Built on `httpx.AsyncClient` with typed contracts (`LLMRequest`, `LLMResponse`, `Usage`) enforced via Pydantic v2.
- **AWS Full Jitter Backoff Algorithm**: Prevents thundering herd retry storms when distributed nodes hit rate limits simultaneously:
  $$\text{Sleep Delay} = \text{Uniform}\left(0, \, \min\left(\text{MaxDelay}, \, \text{BaseDelay} \times 2^{\text{attempt}}\right)\right)$$
- **Circuit-Breaker Fast-Fail Policy**:
  - `429 Rate Limit` & `5xx Server Error`: **Retryable** with exponential backoff and jitter.
  - `4xx Client Error` & `200 Malformed JSON`: **Non-Retryable**; fast-fails immediately on Attempt 1 with zero retry quota waste.
- **Streaming Handshake Safety**: Retries only apply while establishing the initial HTTP connection. Once tokens begin flowing, mid-stream disconnects are surfaced immediately to prevent token duplication to users.
- **Zero-Egress Fault Injection Harness**: Validated across all 6 distributed failure modes using mocked HTTP layers (`respx`).

---

### Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics (`transformer`)

Week 2 connects theoretical Transformer mechanics to empirical measurements, dissecting the physical latency profile of Large Language Model generation.

- **The Two Computational Regimes**:
  1. **Prefill Phase (Prompt Ingestion)**: Ingests all prompt tokens concurrently via matrix multiplications ($QK^T / \sqrt{d_k}$) to populate the Key-Value (KV) cache. Measured client-side as **Time to First Token (TTFT)**. Scales quadratically ($O(N^2)$) with prompt length.
  2. **Decode Phase (Autoregressive Token Generation)**: Iteratively produces one token per forward pass by reading cached KV states from High Bandwidth Memory (HBM). Measured as **Decode Throughput (Tokens/sec)**.
- **Decoding Controls & Output Nondeterminism**:
  - **Temperature ($T$)**: Reshapes the softmax probability distribution over vocabulary logits ($P(x_i) = \frac{\exp(z_i/T)}{\sum \exp(z_j/T)}$). Low temperature ($T \to 0$) collapses toward greedy argmax selection; higher values ($T \ge 0.7$) broaden lexical variation.
  - **Top-P (Nucleus Sampling)**: Truncates candidates to the minimal probability subset whose cumulative sum crosses threshold $p$.
  - **Hardware Floating-Point Nondeterminism**: Demonstrates empirically why modern multi-GPU tensor-parallel inference engines can produce slight output variations even at $T=0.0$ due to non-associative IEEE 754 floating-point reduction across distributed GPU kernels.

---

### Phase 1 Week 3: Schema-Validated Structured Extraction API Service (`robust-api-extractor`)

Week 3 shifts from raw text generation to production-grade structured data extraction, turning unstructured inputs into type-safe, validated domain models.

- **Strict Schema Enforcement**: Extraction schemas defined using Pydantic v2 with field constraints (`Field(ge=0, le=100)`, regex formats, ISO-8601 timestamps, enum literals).
- **Native Tool Calling**: Utilizes OpenAI-compatible JSON Schema tool calling on `openai/gpt-oss-120b` via Groq Cloud.
- **Bounded Self-Correcting Retry Loop**: When the model outputs malformed JSON or violates field constraints:
  1. The client intercepts the `ValidationError` or `JSONDecodeError`.
  2. Constructs an adversarial feedback prompt containing the exact field errors:
     `"Your previous response failed validation: [loc: ('urgency',), msg: 'Input should be 1, 2, or 3']. Please fix this error and call the function again."`
  3. Feeds error history back into the conversation for up to $N=3$ bounded correction attempts.
- **Semantic Sanity Guards**: Post-validation business logic verification (e.g., verifying invoice line item subtotals match the reported total, flagging dates occurring in the past or far future).
- **Token-by-Token SSE Streaming**: Real-time Server-Sent Events endpoint streaming partial JSON tokens to clients while computing final Pydantic validation upon stream termination.
- **Interactive Pipeline Auditor Studio**: Multi-interface web studio featuring live extraction, fault simulation, self-correction traces, and token economics tracking.

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
| FastAPI Gateway (app.py / main.py) & Streamlit Studio (streamlit_demo.py)                         |
|  - Validates request payload against Pydantic schemas (LLMRequest, ExtractRequest)                |
|  - Enforces domain exceptions (LLMRateLimitError, LLMBadRequestError, ExtractionFailure)         |
|  - Streams Server-Sent Events (SSE) token-by-token with monotonic telemetry                      |
|  - Exposes interactive Pipeline Auditor Studio & Fault Injection Simulation Harness              |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Calls client.complete() / extract()
                                                        v
+--------------------------------------------------------------------------------------------------+
| Extraction & Resilient Client Engine (extractor.py & llm_client/client.py)                       |
|  - Decorrelated Full-Jitter Exponential Backoff                                                  |
|  - Fast-fail circuit breaker for non-retryable 4xx / malformed schemas                           |
|  - Bounded Self-Correcting Retry Loop (Injects Pydantic validation errors back to model)         |
|  - Post-validation Semantic Sanity Guards (Line-item reconciliation, date logic)                 |
|  - High-resolution wall-clock monotonic timing via time.perf_counter()                           |
+-------------------------------------------------------+------------------------------------------+
                                                        |
                                                        | Outbound HTTPS / TLS Handshake
                                                        v
+--------------------------------------------------------------------------------------------------+
| Upstream Serving Infrastructure (openai/gpt-oss-120b on Groq Cloud)                              |
|  - Prefill Phase: Subword Tokenization -> Embeddings -> Transformer Attention -> KV Cache Init   |
|  - Decode Phase: Logits -> Softmax Temperature / Top-P Sampling -> Autoregressive Streaming     |
|  - Tool Calling: Structured JSON function-calling arguments emission                             |
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

> **Key Empirical Insight**: On the Long Prompt at `temperature=0.0`, 2 distinct text variations were observed across 3 repeats. This empirically verifies that in distributed GPU serving environments, floating-point reduction non-associativity across concurrent batches can introduce subtle divergences even during greedy sampling.

---

## ⏱️ Weekly Review Protocol & Submission Standards

Every milestone concludes on Friday with a structured **45-minute technical review**:

```
+-----------------------------------------------------------------------------+
| 45-MINUTE WEEKLY TECHNICAL REVIEW BUDGET                                    |
+------------------+--------------------+------------------+------------------+
|     10 min       |       15 min       |      10 min      |      10 min      |
| Concept Defense  | Live Debug / Code  | Benchmark Review | Next Milestone   |
| (No notes)       | Inspection         | & Failure Modes  | Scoping & SLA    |
+------------------+--------------------+------------------+------------------+
```

### Mandatory Deliverables for Every Submission
1. **GitHub Branch & PR**: Clean commit history, descriptive PR explanation.
2. **Automated Tests**: Unit and integration tests with mocked transports (`respx`).
3. **Reproducible Benchmarks**: CLI scripts outputting empirical latency and memory numbers.
4. **Architecture Documentation**: High-level diagrams and failure recovery models.
5. **Tradeoffs & Limitations Report**: Honest technical analysis of constraints.
6. **Live Working Demo**: Terminal CLI or interactive UI proving functionality.

---

## 🎯 Readiness Scorecard & Qualification Standard

Engineers are evaluated on a weighted 100-point scale across 6 core competency areas based strictly on working code and reproducible evidence:

| Competency Area | Weight | Passing Threshold |
| :--- | :---: | :---: |
| **LLM Fundamentals & Networking APIs** | **10%** | Demonstrated in code |
| **RAG Retrieval & Evaluation Systems** | **20%** | Demonstrated in code |
| **Million-Document Architecture & Scaling** | **20%** | $\ge \mathbf{70\%}$ Required |
| **LangChain & LangGraph Workflows** | **15%** | Demonstrated in code |
| **Inference & GPU Memory Optimization** | **20%** | $\ge \mathbf{70\%}$ Required |
| **Production Engineering & Security** | **15%** | $\ge \mathbf{70\%}$ Required |
| **TOTAL WEIGHTED TARGET** | **100%** | $\mathbf{\ge 80 / 100}$ **Overall** |

---

## ⚡ Quickstart & Running Locally

### Prerequisites
- Python 3.10+
- Groq Cloud API Key (`GROQ_API_KEY` or `LLM_API_KEY`)

```bash
# Clone the repository
git clone https://github.com/Rokibul1022/llm_engineer_track.git
cd llm_engineer_track

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

---

### 1. Phase 1 Week 1 & Week 2: Client & Benchmark Studio

```bash
cd phase1-week1/llm-client
pip install -e .

# Run hermetic test suite (11/11 passing in ~0.45s)
pytest tests/ -v

# Run interactive benchmark CLI
python scripts/run_benchmark.py --prompt "Explain attention in 2 sentences."

# Launch dual-phase Streamlit demo studio
streamlit run streamlit_demo.py
```
Open **`http://localhost:8501`** to access:
- **`1_Week1_Fault_Injection`**: Interactive fault injection and physical latency HUD.
- **`2_Week2_Benchmark`**: Live interactive streaming & multi-temperature nondeterminism sweeps.

---

### 2. Phase 1 Week 3: Schema-Validated Extraction Service

```bash
cd ../../phase1-week3
pip install -r requirements.txt

# Start FastAPI extraction service with SSE streaming
uvicorn main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser to inspect the **Pipeline Auditor Studio**:
- Test ticket and invoice schema extraction.
- Simulate schema faults and observe live self-correcting retry feedback loops.
- Inspect semantic sanity check violations and token-by-token SSE streaming.

---

## 📁 Repository Structure

```text
llm_engineer_track/
├── README.md                                  # Master track curriculum, architecture & roadmap
├── answers.md                                 # Core knowledge questions & technical answers
├── architecture.md                            # Transformer inference pipeline specification
├── architecture.png                           # Visual architecture diagram
├── llm-engineer-track.html                    # Interactive 24-week curriculum web application
│
├── phase1-week1/                              # Phase 1 Week 1 & 2: Client & Inference Lab
│   └── llm-client/
│       ├── app.py                             # FastAPI proxy gateway & SSE endpoint
│       ├── streamlit_demo.py                  # Streamlit entrypoint
│       ├── pages/
│       │   ├── 1_Week1_Fault_Injection.py     # Week 1 Live Client & Fault Injection Studio
│       │   └── 2_Week2_Benchmark.py           # Week 2 Token Benchmark & Live Generation Lab
│       ├── llm_client/
│       │   ├── client.py                      # AsyncLLMClient with AWS Full Jitter backoff
│       │   ├── schemas.py                     # Pydantic v2 data models
│       │   ├── exceptions.py                  # Domain exception hierarchy
│       │   └── benchmark.py                   # Latency & nondeterminism benchmark engine
│       ├── scripts/
│       │   └── run_benchmark.py               # Standalone CLI runner
│       └── tests/                             # Hermetic unit tests (respx mocked)
│
├── phase1-week3/                              # Phase 1 Week 3: Structured Extraction Service
│   ├── main.py                                # FastAPI service & SSE streaming endpoint
│   ├── extractor.py                           # Self-correcting retry loop & semantic checks
│   ├── groq_client.py                         # Groq Cloud API caller (tool calling & streaming)
│   ├── schemas.py                             # Pydantic extraction schemas (SupportTicket, Invoice)
│   ├── prompts.py                             # Extraction prompts & tool definitions
│   ├── exceptions.py                          # Extraction failure exceptions
│   ├── prompt.md                              # 13 adversarial evaluation test cases
│   └── static/
│       └── index.html                         # Pipeline Auditor Studio UI
│
└── phase2-week4/                              # Phase 2 Week 4: Document Ingestion (Upcoming)
```

---

## ⚖️ Tradeoffs & Known Limitations

1. **Client-Side API Boundary vs. Model Internal Weights**:
   The client interacts across an external HTTP/SSE API boundary. Token counts and completion latencies reflect provider-reported usage rather than directly instrumented GPU kernel timers.
2. **TTFT Latency Composition**:
   Client-measured TTFT includes client serialization, WAN internet transit, provider API gateway routing, and GPU queueing alongside the true model prefill time.
3. **Provider Rate Limits (RPM/TPM)**:
   Free-tier cloud provider quotas (e.g. Groq 30 RPM) require pacing delays (`--delay 2.1`) during large Cartesian benchmark sweeps to prevent transient HTTP 429 throttling.
4. **Self-Correction Latency & Cost Multiplier**:
   When bounded self-correction retries trigger ($N=3$), each correction attempt sends the accumulated prompt, failed output, and JSON Schema errors back to the model, linearly increasing token consumption and wall-clock turnaround latency.

---

## 🤝 Branch Navigation & Integration

| Milestone | Branch Name | GitHub Pull Request | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1: Week 1** | [`llm-client`](https://github.com/Rokibul1022/llm_engineer_track/tree/llm-client) | [View PR](https://github.com/Rokibul1022/llm_engineer_track/pull/new/llm-client) | Resilient async client, AWS Full Jitter backoff, circuit-breaker fast-fail, zero-egress fault testing. |
| **Phase 1: Week 2** | [`transformer`](https://github.com/Rokibul1022/llm_engineer_track/tree/transformer) | [View PR](https://github.com/Rokibul1022/llm_engineer_track/pull/new/transformer) | Prefill vs decode benchmarking, TTFT decomposition, temperature/top-p nondeterminism sweeps. |
| **Phase 1: Week 3** | [`robust-api-extractor`](https://github.com/Rokibul1022/llm_engineer_track/tree/robust-api-extractor) | [View PR](https://github.com/Rokibul1022/llm_engineer_track/pull/new/robust-api-extractor) | Schema-validated structured extraction, bounded self-correction retry engine, semantic sanity guards, SSE streaming. |

- **Integration Target**: `main`
- **Curriculum Author**: Rokibul
