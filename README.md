# ⚡ Phase 1 Week 2: Inference Pipeline, Token Benchmarking & Decoding Dynamics

> **Transformer Inference Profiling:** Prefill vs Decode Mechanics, Server-Side TTFT Decomposition, KV Cache Dynamics, Temperature/Top-P Nondeterminism Analysis, and Multi-Page Streamlit Studio.

---

## 📑 Table of Contents
1. [Overview & Engineering Objectives](#-overview--engineering-objectives)
2. [Transformer Inference Mechanics: Prefill vs. Decode](#-transformer-inference-mechanics-prefill-vs-decode)
   - [The Two Computational Regimes](#the-two-computational-regimes)
   - [KV Cache Mechanics & Memory Bottlenecks](#kv-cache-mechanics--memory-bottlenecks)
3. [Latency Decomposition & Metrics Breakdown](#-latency-decomposition--metrics-breakdown)
   - [Time to First Token (TTFT)](#1-time-to-first-token-ttft)
   - [Inter-Token Latency (ITL) & Throughput](#2-inter-token-latency-itl--throughput)
4. [Decoding Dynamics & Stochastic Sampling](#-decoding-dynamics--stochastic-sampling)
   - [Temperature Softmax Rescaling](#1-temperature-softmax-rescaling)
   - [Top-P (Nucleus) Probability Mass Truncation](#2-top-p-nucleus-probability-mass-truncation)
   - [Hardware Floating-Point Nondeterminism at T=0.0](#3-hardware-floating-point-nondeterminism-at-t00)
5. [System Architecture & Multi-Page Studio](#-system-architecture--multi-page-studio)
6. [Empirical Benchmark Findings (`openai/gpt-oss-120b`)](#-empirical-benchmark-findings-openaigpt-oss-120b)
7. [Quickstart & Running Locally](#-quickstart--running-locally)

---

## 🎯 Overview & Engineering Objectives

In Phase 1 Week 2, we connect theoretical Transformer mechanics to empirical measurements, dissecting the physical latency and throughput profiles of Large Language Model generation under production workloads.

### Key Objectives:
- **Dissect Transformer Latency**: Distinguish compute-bound prompt prefill from memory-bandwidth-bound autoregressive decoding.
- **High-Resolution Monotonic Instrumentation**: Measure Time to First Token (TTFT), Inter-Token Latency (ITL), and end-to-end roundtrip latency using `time.perf_counter()`.
- **Demystify Nondeterminism**: Quantify how temperature, top-p, and parallel multi-GPU floating-point reduction affect lexical diversity and output reproducibility.
- **Production Interactive Studio**: Deliver a unified multi-page Streamlit application featuring live SSE token streaming, a 5-stage backend execution pipeline stepper, and zero-egress fault injection simulation.

---

## 🔬 Transformer Inference Mechanics: Prefill vs. Decode

### The Two Computational Regimes

Modern autoregressive Transformer inference consists of two fundamentally distinct operational phases:

```
[Prompt Tokens: x₁, x₂, ..., xₙ]
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Prefill (Prompt Ingestion)                   │
│ • Compute-bound: All prompt tokens ingested in parallel│
│ • O(N²) self-attention matrix operations               │
│ • Initializes Key-Value (KV) cache for all tokens      │
│ • Measured on client as Time to First Token (TTFT)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Autoregressive Decode (Token-by-Token)        │
│ • Memory-bandwidth-bound: Iterative single-token passes│
│ • Re-reads cached KV tensors from High Bandwidth Memory│
│ • Projects final hidden state to vocabulary logits     │
│ • Measured as Inter-Token Latency (ITL) & Throughput   │
└────────────────────────────────────────────────────────┘
```

### KV Cache Mechanics & Memory Bottlenecks

In standard Multi-Head Attention (MHA), generating token $t+1$ requires attention over all preceding tokens $1 \dots t$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Without caching, computing $K$ and $V$ for all historical tokens at every step would require $O(T^2)$ compute. The **KV Cache** stores the computed Key and Value projection matrices in GPU High Bandwidth Memory (HBM).

- **Prefill Bottleneck**: Compute-bound (FLOPS limited) by dense tensor contractions.
- **Decode Bottleneck**: Memory-bandwidth-bound (Bytes/sec limited). For each single new token generated, the model must read all historical KV cache entries from HBM across the memory bus, creating an operational compute-to-memory arithmetic intensity of near $O(1)$.

---

## ⏱️ Latency Decomposition & Metrics Breakdown

### 1. Time to First Token (TTFT)
The duration between the client dispatching the HTTP request and receiving the first completion token chunk:
$$\text{TTFT} = t_{\text{DNS}} + t_{\text{TLS}} + t_{\text{Queue}} + t_{\text{Prefill}} + t_{\text{FirstChunkTransport}}$$

**Empirical Observation:** TTFT scales with prompt token count. Under high server load, queueing latency can dominate over raw GPU prefill time.

### 2. Inter-Token Latency (ITL) & Throughput
The average time required to generate each subsequent token once decoding has started:
$$\text{ITL} = \frac{t_{\text{FinalToken}} - t_{\text{FirstToken}}}{N_{\text{DecodeTokens}} - 1}$$
$$\text{Throughput} = \frac{N_{\text{DecodeTokens}}}{t_{\text{DecodeDuration}}} \quad (\text{tokens/sec})$$

---

## 🎲 Decoding Dynamics & Stochastic Sampling

### 1. Temperature Softmax Rescaling
Temperature $T$ modulates the sharpness of the probability distribution over vocabulary logits $z_i$:
$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **$T \to 0$ (Greedy Decoding)**: Collapses into an argmax choice. Picks the highest probability token deterministically.
- **$T \in [0.7, 1.2]$**: Flattens the distribution, giving lower-ranked tokens higher selection probability and increasing creativity.
- **$T > 1.5$**: Flattens logits excessively, leading to semantic incoherence and syntax degradation.

### 2. Top-P (Nucleus) Probability Mass Truncation
Rather than sampling across the entire vocabulary, Top-P restricts candidate tokens to the smallest subset $V^{(p)}$ whose cumulative probability exceeds threshold $p$:
$$\sum_{x_i \in V^{(p)}} P(x_i) \ge p$$

Unlike fixed Top-K, Top-P dynamically expands when the model is uncertain (flat distribution) and contracts to 1–2 candidates when the model is highly confident (steep distribution).

### 3. Hardware Floating-Point Nondeterminism at T=0.0
Even with $T=0.0$, multi-GPU tensor parallel inference engines (e.g. vLLM, TensorRT-LLM, Groq LPU clusters) can produce minor output differences across identical runs. 

**Root Cause:** IEEE 754 floating-point addition is non-associative:
$$(A + B) + C \neq A + (B + C)$$
In distributed all-reduce operations across GPU threads and warp blocks, the order in which partial sums arrive depends on non-deterministic hardware scheduling, occasionally causing ties at the argmax logit boundary to flip.

---

## 🏛️ System Architecture & Multi-Page Studio

```
┌────────────────────────────────────────────────────────┐
│     Interactive Streamlit Studio (streamlit_demo.py)   │
│  • Visual 5-Stage Stepper: Ingestion ➔ Transport ➔     │
│    Prefill & TTFT ➔ Decode Loop ➔ Billing Ledger       │
│  • Latency & Cost HUD (Decomposes TTFT vs ITL)         │
│  • Fault Injection Simulator (429, 5xx, 4xx, 200 OK)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           │ Async Calls
                           ▼
┌────────────────────────────────────────────────────────┐
│        AsyncLLMClient Core (llm_client/client.py)      │
│  • Pydantic v2 Contract Validation (LLMRequest)        │
│  • AWS Full-Jitter Exponential Backoff                 │
│  • Fast-Fail Circuit Breaker Policy                    │
│  • Server-Sent Events (SSE) Streaming Generator        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           │ HTTPS TLS 1.3
                           ▼
┌────────────────────────────────────────────────────────┐
│         Upstream Inference Engine (Groq Cloud)         │
│  • Model: openai/gpt-oss-120b                          │
│  • Low-latency LPUs with ultra-fast token streaming    │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmark Findings (`openai/gpt-oss-120b`)

Extracted from empirical benchmark sweeps across prompt lengths and temperature configurations:

| Prompt Regime | Input Tokens | Output Tokens | TTFT (ms) | Decode Throughput | Output Behavior |
|---|:---:|:---:|:---:|:---:|---|
| **Short (20 tok)** | 22 | 48 | ~220 ms | ~110 tok/sec | Stable, concise definitions |
| **Medium (150 tok)**| 154 | 120 | ~340 ms | ~105 tok/sec | Detailed structural answers |
| **Long (500 tok)** | 512 | 240 | ~680 ms | ~98 tok/sec | In-depth technical breakdown |
| **Greedy ($T=0.0$)**| 22 | 50 | ~215 ms | ~112 tok/sec | 100% lexical match across runs |
| **Creative ($T=1.0$)**| 22 | 65 | ~230 ms | ~108 tok/sec | High vocabulary diversity |

---

## 🚀 Quickstart & Running Locally

### 1. Environment Configuration
Verify your `.env` file in `phase1-week1/llm-client/`:
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_DEFAULT_MODEL=openai/gpt-oss-120b
```

### 2. Run the Automated Unit Test Suite
```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\pytest.exe" .\phase1-week1\llm-client\tests\ -v
```

### 3. Launch the Multipage Interactive Studio
```powershell
& ".\phase1-week1\llm-client\.venv\Scripts\streamlit.exe" run ".\phase1-week1\llm-client\streamlit_demo.py"
```

Open **`http://localhost:8501/`** to explore:
- **Tab 1: Live Client & Speed Insights**: Real-time TTFT vs. inter-token latency decomposition.
- **Tab 2: Acceptance & Fault Injection Studio**: Interactive simulations of rate limits (429), server crashes (5xx), malformed payloads, and circuit breaker fast-fails.
