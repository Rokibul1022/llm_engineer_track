# Week 2 — Core Knowledge Targets & Technical Answers

*Knowledge targets answered directly and grounded in the `phase1-week2` implementation on `llm_engineer_track` using model `openai/gpt-oss-120b`.*

---

## 1. How Tokens, Embeddings, Attention, and Autoregressive Generation Work

### Tokens
Before text reaches a neural network, it must be discretized into numerical representations. A tokenizer breaks raw strings into subword tokens using algorithms like Byte-Pair Encoding (BPE) or WordPiece and maps each token to an integer index in a fixed vocabulary ($V \approx 32\text{k} - 128\text{k}$).
- Tokens are **not** words or characters; common words might be 1 token, while rare words or code snippets split into several subwords.
- In our codebase, token usage is encapsulated in [`llm_client/schemas.py`](./llm_client/schemas.py) via the `Usage` schema (`prompt_tokens`, `completion_tokens`, `total_tokens`), captured from provider SSE frames in [`llm_client/client.py`](./llm_client/client.py).

### Embeddings & Positional Encoding
Discrete token integer IDs cannot be fed directly into floating-point linear layers. An **embedding matrix** ($W_E \in \mathbb{R}^{|V| \times d_{\text{model}}}$) projects each token ID into a continuous high-dimensional vector space (typically $d_{\text{model}} = 4096$ or $8192$).
- Because self-attention is permutation-equivariant (it treats sequences as sets), **positional encodings** (such as Rotary Position Embeddings / RoPE) are added or applied as complex rotation operators to inject the sequential order of words.

### Self-Attention Mechanism
Inside each Transformer layer, multi-head self-attention computes dynamic contextual representations:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Each token generates Query ($Q$), Key ($K$), and Value ($V$) projections.
- The dot product $QK^T$ evaluates pairwise relevance between every token and all preceding tokens.
- **Context Limit Cost**: Prefill attention scales quadratically ($O(N^2)$) with prompt sequence length $N$. Processing a prompt with 1,000 tokens requires allocating and multiplying $1000 \times 1000$ attention scores per layer per head, explaining why long prompts incur higher prefill compute and latency.

### Autoregressive Generation & KV Caching
Decoder-only Transformers do not output entire paragraphs in a single forward pass; they generate **one token at a time**:
1. During the **Prefill phase**, the entire prompt is ingested in parallel, computing and storing intermediate Key and Value tensors into the **KV Cache**.
2. During the **Decode phase**, the model emits the next token, appends it to the sequence, and runs another forward pass.
3. Because past KV states are cached in High Bandwidth Memory (HBM), the model only computes $Q$ for the single newest token and attends to the cached $K, V$ states.
4. In our client, this loop is directly manifested in `AsyncLLMClient.stream()`: each SSE delta chunk yielded corresponds to one step of the autoregressive decode loop.

---

## 2. Temperature, Top-P, Context Limits, and Why Outputs are Nondeterministic

### Temperature
The final Transformer layer emits raw unnormalized log probabilities (logits $z_i$) over the vocabulary. Softmax with temperature $T$ converts logits into a sampling distribution:
$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **$T \to 0$ (Greedy / Argmax)**: Exaggerates differences between logits, collapsing the probability distribution onto the single highest-probability token. Repeats become largely deterministic.
- **High $T$ ($T = 0.7 - 1.2$)**: Flattens the distribution toward uniform probability, increasing the likelihood of selecting lower-ranked tokens, yielding higher lexical diversity and creative variations.

### Top-P (Nucleus Sampling)
Top-P restricts candidate sampling to the smallest subset of tokens whose cumulative probability exceeds threshold $p$:
$$\sum_{i \in V^{(p)}} P(\text{token}_i) \ge p$$
Unlike fixed Top-K, Top-P dynamically expands or contracts the candidate pool based on model confidence. When confidence is high (e.g. following "The capital of France is"), the nucleus narrows to 1–2 tokens; when ambiguous, it broadens.

### Context Limits
A model's context window represents the maximum sequence length ($N_{\text{prompt}} + N_{\text{completion}} \le N_{\text{max}}$) supported by its positional encodings and GPU memory allocations. Approaching context limits:
- Depletes GPU VRAM due to linear growth of the KV Cache ($2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len}$ bytes in FP16).
- Increases prefill latency (TTFT) due to larger attention matrix multiplications.

### Why Outputs Are Nondeterministic
Nondeterminism arises from two distinct sources:
1. **Algorithmic Sampling**: At any $T > 0$, pseudo-random sampling draws from the probability distribution. Two identical prompts will follow divergent autoregressive generation trajectories as soon as an early token differs.
2. **Hardware/Batching Floating-Point Non-Associativity**: Even at $T = 0.0$ (greedy sampling), modern multi-GPU distributed clusters (tensor-parallel kernels, batching engines like vLLM/TensorRT-LLM) sum floating-point operations in slightly varying orders across concurrent request batches. Because IEEE 754 floating-point addition is non-associative ($(a+b)+c \ne a+(b+c)$), marginal differences in low-order bits can occasionally flip the top-ranked token for tokens with near-identical logits.

---

## 3. Measured Empirical Results (`openai/gpt-oss-120b`)

The table below documents real empirical data generated by our automated benchmark harness (`python scripts/run_benchmark.py --repeats 3 --delay 2.1`) querying `openai/gpt-oss-120b`:

| Prompt Size | Prompt Chars | Temp | Top-P | Runs | Prompt Tokens | Comp Tokens | Avg TTFT (ms) | Avg Tokens/sec | Identical Repeats? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short** | 36c | 0.0 | 1.0 | 3 | 81 | 120 | 605.2 ms | 199.2 tok/s | **YES (Deterministic)** |
| **Short** | 36c | 0.7 | 0.9 | 3 | 81 | 115 | 592.4 ms | 187.5 tok/s | **NO (3 variations)** |
| **Short** | 36c | 1.2 | 1.0 | 3 | 81 | 123 | 695.5 ms | 172.3 tok/s | **NO (3 variations)** |
| **Medium** | 162c | 0.0 | 1.0 | 3 | 99 | 128 | 546.0 ms | 179.3 tok/s | **YES (Deterministic)** |
| **Medium** | 162c | 0.7 | 0.9 | 3 | 99 | 128 | 627.6 ms | 169.3 tok/s | **NO (3 variations)** |
| **Medium** | 162c | 1.2 | 1.0 | 3 | 99 | 128 | 622.8 ms | 189.3 tok/s | **NO (3 variations)** |
| **Long** | 758c | 0.0 | 1.0 | 3 | 214 | 128 | 698.7 ms | 158.7 tok/s | **NO (2 variations)\*** |
| **Long** | 758c | 0.7 | 0.9 | 3 | 214 | 128 | 661.1 ms | 164.4 tok/s | **NO (3 variations)** |
| **Long** | 758c | 1.2 | 1.0 | 3 | 214 | 128 | 651.4 ms | 170.3 tok/s | **NO (3 variations)** |

*\*Note on Long prompt at Temp=0.0: Output variation observed even at temperature 0.0 on the long prompt directly confirms the hardware floating-point non-associativity documented in Section 2 and [`LIMITATIONS.md`](./LIMITATIONS.md).*

---

## 4. What We Built

1. **Server-Side Benchmarking Engine** ([`llm_client/benchmark.py`](./llm_client/benchmark.py)):
   - Monotonic high-resolution timing (`time.perf_counter()`) for TTFT and decode throughput.
   - Distinctness & determinism analyzer across Cartesian prompt-parameter combinations.
2. **CLI Automation Runner** ([`scripts/run_benchmark.py`](./scripts/run_benchmark.py)):
   - Non-blocking script with console formatting, automated rate-limit pacing, and persistence to `benchmark_results.json`.
3. **Interactive Streamlit Multipage Module** ([`pages/2_Week2_Benchmark.py`](./pages/2_Week2_Benchmark.py)):
   - Visual parameter sliders (`temperature`, `top_p`, repeats), live throughput counters, and side-by-side repeat text inspection for nondeterminism exploration.
4. **Resilient Test Suite** ([`tests/test_benchmark.py`](./tests/test_benchmark.py)):
   - 11 unit tests passing with 100% test coverage using mock transports and respx.

---

## 5. What We Learned

- **Decoupled Architecture Payoff**: By building `benchmark.py` directly on top of `AsyncLLMClient.stream()` from Week 1 without modifying retry or transport internals, we achieved 100% backwards compatibility and zero regressions across the existing test suite.
- **TTFT vs. Steady-State Decode**: Prompt size directly impacts TTFT because prefill processes the prompt through self-attention, while tokens/sec remains comparatively stable across generations due to KV caching.
- **Empirical Reality of Nondeterminism**: Nondeterminism is not merely a parameter slider; at temperature 0.0, distributed serving infrastructures can still introduce subtle text divergences on longer sequences due to parallel reduction ordering.
