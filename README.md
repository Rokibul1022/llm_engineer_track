# 🚀 LLM Engineer Track: From Async Python to Production Agents & RAG

> **An end-to-end, production-grade LLM engineering initiative comprising 24 progressive technical milestones**—advancing from core asynchronous Python architectures and resilient streaming clients to autonomous multi-agent systems and retrieval-augmented generation (RAG) platforms.

---

## 🗺️ Roadmap & Curriculum (24 Technical Tasks)

| Phase | Milestone / Focus | Status | Key Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1: Week 1** | **Resilient Async LLM Client & Latency Insights** | ✅ Completed | Async client, Full Jitter backoff, fast-fail circuit breaker, TTFT & decode decomposition, live acceptance testing harness. |
| **Phase 1: Week 2** | **Inference Pipeline & Token Benchmarking** | ✅ Completed | Transformer pipeline explanation, server-side TTFT & throughput benchmark, temperature/top-p nondeterminism analysis, multipage Streamlit UI. |

| **Phase 1: Week 3** | **Streaming Architectures & SSE Protocol** | 📋 Scheduled | Real-time token pipelines, backpressure handling, bi-directional event transport. |
| **Phase 1: Week 4** | **Observability, Tracing & Token Economics** | 📋 Scheduled | OpenTelemetry integration, distributed trace propagation, granular per-token cost ledger. |
| **Phase 2** | **Retrieval-Augmented Generation (RAG)** | 📋 Scheduled | Hybrid search (dense + BM25), reciprocal rank fusion, dynamic reranking, chunking optimization. |
| **Phase 3** | **Production Multi-Agent Systems & Tool Use** | 📋 Scheduled | Planning engines, human-in-the-loop validation, sandbox execution, agentic workflows. |
| **Phase 4** | **Enterprise Evaluation, Guardrails & Deployment** | 📋 Scheduled | Automated red teaming, latency SLAs, Kubernetes deployment, circuit breakers at scale. |

---

## 📂 Active Phase: Phase 1 Week 2 — Inference Pipeline Explanation & Token Benchmark

Explore the implementation in [`phase1-week1/llm-client/`](./phase1-week1/llm-client/):

* **[Phase 1 Week 1 & 2 LLM Client (`phase1-week1/llm-client/`)](./phase1-week1/llm-client/)**:
  * Production-grade `AsyncLLMClient` with exponential backoff & full jitter
  * Circuit-breaker fast-fail policy for 4xx and malformed payloads
  * **[Inference Pipeline Deep Dive (`PIPELINE.md`)](./phase1-week1/llm-client/PIPELINE.md)**: Tokenize, embed, attention, sample, decode loop mapped to codebase.
  * **[System Architecture (`architecture.md`)](./architecture.md)** & Architecture Diagram (`architecture.png`).
  * **[Core Knowledge Answers (`answers.md`)](./answers.md)**: Full answers on tokenization, embeddings, attention, autoregressive decoding, temperature, top-p, context limits, and nondeterminism.
  * **[Tradeoffs & Limitations (`LIMITATIONS.md`)](./phase1-week1/llm-client/LIMITATIONS.md)**: Client-side boundary, TTFT latency conflation, empirical determinism bounds.
  * **Automated Benchmark Suite (`scripts/run_benchmark.py`)**: Monotonic TTFT, tokens/sec, and temperature sweeps.
  * **Interactive Multipage Streamlit Studio**: Week 1 Fault Injection (`pages/1_Week1_Fault_Injection.py`) and Week 2 Benchmark Explorer (`pages/2_Week2_Benchmark.py`).

