# Tradeoffs & Known Limitations

This document outlines key technical tradeoffs, architectural boundaries, and known limitations of the Week 2 benchmarking harness and inference client.

---

### 1. Client-Side API Boundary vs. Internal Model Observability
- **Limitation**: This codebase operates strictly across an external HTTP/REST/SSE boundary targeting `openai/gpt-oss-120b` (hosted on Groq).
- **Impact**: Model internal weights, embedding projections, attention score matrices ($QK^T$), and raw logit distributions are completely inaccessible from the client. Token counts (`prompt_tokens`, `completion_tokens`) reflect provider-reported usage rather than locally verified tokenizer encodings.

### 2. Time to First Token (TTFT) Conflation (Network & Queueing vs. Compute)
- **Limitation**: `ttft_s` is measured client-side from request dispatch (`time.perf_counter()`) to the arrival of the first SSE chunk (`delta.content`).
- **Impact**: Measured TTFT does **not** solely represent prompt prefill compute. It conflates:
  1. Client-side serialization and TLS connection handshake.
  2. WAN internet round-trip latency (RTT) between the client and provider endpoints.
  3. Provider-side gateway scheduling, load balancing, and GPU queueing delays.
  4. Actual model forward-pass prefill latency.

### 3. Empirical Nondeterminism and Sample Size Limitations
- **Limitation**: The benchmark tests $N=3$ repeats per parameter configuration.
- **Impact**:
  - Observing identical text outputs across 3 runs at `temperature=0.0` strongly suggests deterministic greedy sampling, but does not provide mathematical proof of determinism.
  - Furthermore, modern distributed inference clusters exhibit non-deterministic floating-point accumulation orders across tensor-parallel GPU kernels. As observed in our live benchmarks on the **Long prompt at `temperature=0.0`**, 2 distinct variations were observed despite greedy sampling settings. Nondeterminism is therefore an empirical observation rather than an absolute binary state.

### 4. Provider Rate Limits & Pacing Overhead
- **Limitation**: Cloud LLM endpoints enforce strict Requests Per Minute (RPM) and Tokens Per Minute (TPM) quotas (e.g., Groq on-demand free tier limits requests to 30 RPM).
- **Impact**: Running large Cartesian sweeps ($3 \text{ prompts} \times 3 \text{ settings} \times 3 \text{ repeats} = 27 \text{ requests}$) without throttling quickly triggers HTTP 429 rate limit exceptions. We introduced `delay_between_runs_s: float = 2.0` in `run_benchmark_suite()` and `scripts/run_benchmark.py` to pace execution, which extends total suite benchmark duration to ~60–80 seconds.

### 5. Streamlit Concurrency Model (`asyncio.run`)
- **Limitation**: Streamlit's reactive execution model re-executes Python scripts from top to bottom on each interaction, executing on synchronous worker threads.
- **Impact**: Inside [`pages/2_Week2_Benchmark.py`](./pages/2_Week2_Benchmark.py), calling `asyncio.run(execute_benchmark())` blocks the active Streamlit session thread until the entire suite or run completes. While acceptable for interactive lab experiments, production benchmarking at enterprise scale requires decoupled background workers (e.g. Celery / Redis Queue) with WebSocket status updates.
