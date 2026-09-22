"""Week 2 Module: LLM Token Generation Benchmark & Nondeterminism Explorer.

Interactive UI for benchmarking TTFT (Time to First Token), decode throughput (tokens/sec),
and observing output nondeterminism across temperature & top-p settings.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import httpx
import streamlit as st

from llm_client import (
    AsyncLLMClient,
    ChatMessage,
    LLMRequest,
    run_benchmark_suite,
    run_single_benchmark,
    summarize,
)
from scripts.run_benchmark import load_env_file

# Load environment configuration
load_env_file(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="Week 2: Token Benchmark & Nondeterminism",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Week 2 — Token Generation Benchmark & Decoding Lab")
st.caption(
    "Measures **Time to First Token (TTFT)**, **decode throughput (tokens/sec)**, and demonstrates "
    "**output nondeterminism** across varying `temperature` and `top_p` sampling configurations."
)

st.markdown("---")

# Configuration Sidebar
with st.sidebar:
    st.header("🔧 Execution Settings")
    default_model = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b")
    model_name = st.text_input("Model ID", value=default_model, disabled=True)

    has_api_key = bool(os.getenv("LLM_API_KEY"))
    exec_mode = st.radio(
        "Execution Transport",
        options=["Live API (Groq / Provider)", "Simulated Mock Transport"],
        index=0 if has_api_key else 1,
        help="Simulated transport generates instant synthetic tokens without spending rate limits.",
    )
    is_mock = "Mock" in exec_mode

    st.markdown("---")
    st.markdown("### 📚 Reference Documents")
    st.markdown("- [`PIPELINE.md`](./PIPELINE.md): Pipeline Architecture")
    st.markdown("- [`ANSWERS.md`](./ANSWERS.md): Core Conceptual Questions")
    st.markdown("- [`LIMITATIONS.md`](./LIMITATIONS.md): Tradeoffs & Gotchas")

# Benchmark Form
col_left, col_right = st.columns([3, 2])

with col_left:
    prompt_input = st.text_area(
        "Prompt",
        value=(
            "Explain what a token is in 10 words and why time-to-first-token differs from subsequent decode latency."
        ),
        height=110,
        help="Input prompt to benchmark against the autoregressive decoding pipeline.",
    )

with col_right:
    mode_selection = st.selectbox(
        "Benchmark Mode",
        [
            "Custom Setting (Single Config × Repeats)",
            "Temperature Comparison Sweep (0.0, 0.7, 1.2 × Repeats)",
        ],
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1,
            disabled=("Sweep" in mode_selection),
        )
    with col_t2:
        top_p = st.slider("Top-P (Nucleus)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    repeats = st.number_input("Repeats per configuration", min_value=1, max_value=5, value=3)

run_button = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)


def build_mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][-1]["content"]
        temp = body.get("temperature", 0.7)

        if temp == 0.0:
            msg = f"Deterministic response: Tokens are atomic subword units used by LLMs ({content[:15]}...)"
        else:
            import uuid
            msg = f"Creative sample [{uuid.uuid4().hex[:5]}]: Tokens split language into vectorizable pieces ({content[:15]}...)"

        words = msg.split()
        lines = []
        for w in words:
            lines.append(f'data: {{"choices": [{{"delta": {{"content": " {w}"}}}}]}}\n\n')
        lines.append(f'data: {{"usage": {{"prompt_tokens": {len(content.split()) + 6}, "completion_tokens": {len(words)}, "total_tokens": {len(content.split()) + 6 + len(words)}}}}}\n\n')
        lines.append("data: [DONE]\n\n")

        return httpx.Response(200, text="".join(lines), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


async def execute_benchmark():
    api_key = os.getenv("LLM_API_KEY", "demo-key")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    transport = build_mock_transport() if is_mock else None

    client = AsyncLLMClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,
        base_delay_s=0.5,
        max_delay_s=3.0,
        transport=transport,
    )

    if "Sweep" in mode_selection:
        settings = [
            {"model": default_model, "temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
            {"model": default_model, "temperature": 0.7, "top_p": top_p, "max_tokens": 128},
            {"model": default_model, "temperature": 1.2, "top_p": 1.0, "max_tokens": 128},
        ]
    else:
        settings = [
            {"model": default_model, "temperature": temperature, "top_p": top_p, "max_tokens": 128}
        ]

    pace_delay = 0.0 if is_mock else 1.5

    with st.spinner("Streaming tokens and measuring timing telemetry..."):
        try:
            runs = await run_benchmark_suite(
                client=client,
                prompts=[prompt_input],
                settings=settings,
                repeats=int(repeats),
                delay_between_runs_s=pace_delay,
            )
        finally:
            await client.aclose()

    return runs


if run_button:
    st.markdown("### 📊 Benchmark Results")

    try:
        benchmark_runs = asyncio.run(execute_benchmark())
    except Exception as exc:
        st.error(f"Benchmark run failed: {exc}")
    else:
        summary = summarize(benchmark_runs)
        groups = summary["groups"]

        # Metric cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        avg_ttft_all = sum(g["avg_ttft_ms"] for g in groups) / len(groups)
        avg_tps_all = sum(g["avg_tokens_per_second"] for g in groups) / len(groups)
        total_runs_count = summary["total_runs"]

        m_col1.metric("Average TTFT", f"{avg_ttft_all:.1f} ms")
        m_col2.metric("Avg Throughput", f"{avg_tps_all:.1f} tok/s")
        m_col3.metric("Total Runs", f"{total_runs_count}")
        m_col4.metric(
            "Transport",
            "Mock Transport" if is_mock else "Live API",
        )

        # Summary Table
        table_rows = []
        for g in groups:
            table_rows.append(
                {
                    "Prompt Length": f"{g['prompt_char_length']} chars",
                    "Temperature": f"{g['temperature']:.1f}",
                    "Top-P": f"{g['top_p']}" if g["top_p"] is not None else "default",
                    "Repeats": g["repeats"],
                    "Prompt Tokens": f"{g['avg_prompt_tokens']:.0f}",
                    "Completion Tokens": f"{g['avg_completion_tokens']:.0f}",
                    "TTFT (ms)": f"{g['avg_ttft_ms']:.1f}",
                    "Tokens / sec": f"{g['avg_tokens_per_second']:.1f}",
                    "Identical Across Repeats?": (
                        "✅ YES (Deterministic)"
                        if g["identical_across_repeats"]
                        else f"❌ NO ({g['distinct_outputs_count']} distinct outputs)"
                    ),
                }
            )

        st.dataframe(table_rows, use_container_width=True)

        # Output text inspections for nondeterminism
        st.markdown("#### 🔬 Detailed Output Text per Repeat (Nondeterminism Inspector)")
        st.caption(
            "Compare text variations across runs. At `temperature=0.0`, text outputs converge into identical strings. "
            "At `temperature >= 0.7`, probability sampling yields distinct phrasing variations across runs."
        )

        for g in groups:
            label = (
                f"🌡️ Temperature {g['temperature']:.1f} (Top-P: {g['top_p']}) — "
                + ("Identical outputs" if g["identical_across_repeats"] else f"{g['distinct_outputs_count']} distinct outputs")
            )
            with st.expander(label, expanded=True):
                for idx, out in enumerate(g["outputs"], 1):
                    st.markdown(f"**Run #{idx}:**")
                    st.code(out, language=None)

st.markdown("---")

# Pedagogical summary section as requested
st.markdown("### 📖 What this page demonstrates")
st.markdown(
    """
1. **Inference Pipeline**: Text enters through **Tokenization** (subwords $\\to$ token IDs), gets projected into dense **Embeddings** with positional signals, passes through multi-head **Self-Attention** blocks, produces vocabulary **Logits**, and undergoes **Sampling** before being detokenized.
2. **Autoregressive Decoding**: Each forward pass produces exactly one token. The first token latency (**TTFT**) includes full prompt ingestion (prefill), whereas subsequent tokens leverage the **KV Cache** for fast decode steps.
3. **Sampling Controls & Nondeterminism**:
   - `temperature=0.0` collapses the distribution toward the argmax token, yielding deterministic or near-deterministic outputs across repeats.
   - Higher temperature ($\ge 0.7$) and `top_p` nucleus sampling broaden candidate choices, producing distinct, varied phrasing for identical inputs.
4. **Context Window & Attention Tradeoffs**: Longer prompts require quadratic attention matrix allocations during prefill, visibly increasing TTFT.

*(For complete technical details, see [`PIPELINE.md`](./PIPELINE.md), [`LIMITATIONS.md`](./LIMITATIONS.md), and [`ANSWERS.md`](./ANSWERS.md)).*
"""
)
