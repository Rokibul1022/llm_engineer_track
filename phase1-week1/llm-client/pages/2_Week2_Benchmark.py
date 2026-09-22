"""Week 2 Module: LLM Token Generation Benchmark & Nondeterminism Explorer.

Interactive UI for live real-time token streaming, TTFT (Time to First Token) measurement,
decode throughput (tokens/sec), and output nondeterminism observation across temperature/top-p settings.
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
    Usage,
    run_benchmark_suite,
    summarize,
)
from scripts.run_benchmark import load_env_file

# Load environment configuration
load_env_file(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="Week 2: Token Benchmark & Live Generation",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Week 2 — Live Token Generation & Benchmarking Lab")
st.caption(
    "Interactive studio powered by `AsyncLLMClient`. Test live streaming generation in real-time, "
    "measure **Time to First Token (TTFT)**, **decode throughput (tokens/sec)**, and inspect "
    "**output nondeterminism** across varying `temperature` and `top_p` sampling configurations."
)

st.markdown("---")

# Configuration Sidebar
with st.sidebar:
    st.header("🔧 Engine Configuration")
    default_model = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b")
    model_name = st.text_input("Target Model", value=default_model, disabled=True)

    has_api_key = bool(os.getenv("LLM_API_KEY"))
    exec_mode = st.radio(
        "Execution Transport",
        options=["Live API (Groq Cloud)", "Simulated Mock Transport"],
        index=0 if has_api_key else 1,
        help="Simulated transport generates instant synthetic tokens without spending rate limits.",
    )
    is_mock = "Mock" in exec_mode

    st.markdown("---")
    st.markdown("### 📚 Reference Documents")
    st.markdown("- [`PIPELINE.md`](./PIPELINE.md): Pipeline Architecture")
    st.markdown("- [`ANSWERS.md`](./ANSWERS.md): Core Conceptual Questions")
    st.markdown("- [`LIMITATIONS.md`](./LIMITATIONS.md): Tradeoffs & Gotchas")


def build_mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        content = body["messages"][-1]["content"]
        temp = body.get("temperature", 0.7)

        if temp == 0.0:
            msg = f"Deterministic response: Tokens represent atomic subword units in LLMs. ({content[:20]}...)"
        else:
            import uuid
            msg = f"Creative sample [{uuid.uuid4().hex[:5]}]: Language models vectorize subwords into embeddings. ({content[:20]}...)"

        words = msg.split()
        lines = []
        for w in words:
            lines.append(f'data: {{"choices": [{{"delta": {{"content": " {w}"}}}}]}}\n\n')
        lines.append(f'data: {{"usage": {{"prompt_tokens": {len(content.split()) + 6}, "completion_tokens": {len(words)}, "total_tokens": {len(content.split()) + 6 + len(words)}}}}}\n\n')
        lines.append("data: [DONE]\n\n")

        return httpx.Response(200, text="".join(lines), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


tab_live, tab_benchmark = st.tabs([
    "💬 1. Live Interactive Streaming (Test in Real-Time)",
    "📊 2. Nondeterminism & Multi-Repeat Benchmark Suite",
])

# ==============================================================================
# TAB 1: LIVE STREAMING PLAYGROUND
# ==============================================================================
with tab_live:
    st.subheader("💬 Test Live Generation with Real-Time Telemetry")
    st.caption("Type any prompt below, hit **Generate Response**, and watch tokens stream live with instantaneous TTFT and throughput metrics.")

    col_input, col_params = st.columns([3, 1])

    with col_input:
        live_prompt = st.text_area(
            "Enter your prompt:",
            value="Explain what attention does in Transformer models in 2 concise sentences.",
            height=110,
            key="live_prompt_input",
        )

    with col_params:
        live_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="live_temp")
        live_top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05, key="live_top_p")
        live_max_tokens = st.number_input("Max Tokens", 16, 1024, 256, 32, key="live_max_tokens")

    live_button = st.button("⚡ Generate Response (Live Stream)", type="primary", use_container_width=True)

    if live_button:
        metrics_placeholder = st.empty()
        st.markdown("#### 📝 Generated Response (Live Stream):")
        response_placeholder = st.empty()

        async def stream_live_completion():
            api_key = os.getenv("LLM_API_KEY", "demo-key")
            base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
            transport = build_mock_transport() if is_mock else None

            client = AsyncLLMClient(
                api_key=api_key,
                base_url=base_url,
                max_retries=2,
                transport=transport,
            )

            req = LLMRequest(
                model=default_model,
                messages=[ChatMessage(role="user", content=live_prompt)],
                temperature=live_temp,
                top_p=live_top_p,
                max_tokens=int(live_max_tokens),
                stream=True,
            )

            start_t = time.perf_counter()
            first_t: float | None = None
            collected_text = ""
            token_count = 0
            captured_usage: Usage | None = None

            def on_u(u: Usage):
                nonlocal captured_usage
                captured_usage = u

            try:
                async for chunk in client.stream(req, on_usage=on_u):
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    token_count += 1
                    collected_text += chunk

                    # Update live text stream
                    response_placeholder.markdown(collected_text + " ▌")

                    # Update live metrics
                    current_elapsed = now - start_t
                    current_ttft_ms = (first_t - start_t) * 1000 if first_t else current_elapsed * 1000
                    current_tps = (token_count / current_elapsed) if current_elapsed > 0 else 0

                    with metrics_placeholder.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Time to First Token (TTFT)", f"{current_ttft_ms:.1f} ms")
                        m2.metric("Speed (Throughput)", f"{current_tps:.1f} tok/s")
                        m3.metric("Tokens Generated", f"{token_count}")
                        m4.metric("Elapsed", f"{current_elapsed:.2f} s")

                # Remove cursor
                response_placeholder.markdown(collected_text)

                end_t = time.perf_counter()
                total_duration = end_t - start_t
                final_ttft_ms = (first_t - start_t) * 1000 if first_t else total_duration * 1000
                final_tokens = captured_usage.completion_tokens if (captured_usage and captured_usage.completion_tokens > 0) else token_count
                final_prompt_toks = captured_usage.prompt_tokens if (captured_usage and captured_usage.prompt_tokens > 0) else len(live_prompt.split())
                final_tps = (final_tokens / total_duration) if total_duration > 0 else 0

                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("⏱️ Final TTFT", f"{final_ttft_ms:.1f} ms")
                    m2.metric("⚡ Generation Speed", f"{final_tps:.1f} tok/s")
                    m3.metric("🔢 Total Tokens", f"{final_tokens + final_prompt_toks}", f"{final_prompt_toks} in / {final_tokens} out")
                    m4.metric("⏱️ Total Time", f"{total_duration:.2f} s")

            except Exception as exc:
                st.error(f"Generation error: {exc}")
            finally:
                await client.aclose()

        asyncio.run(stream_live_completion())


# ==============================================================================
# TAB 2: BENCHMARK SUITE & NONDETERMINISM EXPLORER
# ==============================================================================
with tab_benchmark:
    st.subheader("📊 Nondeterminism & Multi-Repeat Benchmark")
    st.caption("Runs prompt sweeps across temperatures (0.0, 0.7, 1.2) to evaluate TTFT scaling and demonstrate output text nondeterminism.")

    b_col_left, b_col_right = st.columns([3, 2])

    with b_col_left:
        bench_prompt = st.text_area(
            "Benchmark Prompt",
            value="Explain what a token is in 10 words and why time-to-first-token differs from subsequent decode latency.",
            height=90,
            key="bench_prompt_input",
        )

    with b_col_right:
        bench_mode = st.selectbox(
            "Parameter Sweep Mode",
            [
                "Temperature Sweep (0.0, 0.7, 1.2 × Repeats)",
                "Custom Single Setting × Repeats",
            ],
            key="bench_mode",
        )
        b_col_s1, b_col_s2 = st.columns(2)
        with b_col_s1:
            b_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, disabled=("Sweep" in bench_mode), key="b_temp")
        with b_col_s2:
            b_top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05, key="b_top_p")
        b_repeats = st.number_input("Repeats per setting", 1, 5, 3, key="b_repeats")

    bench_run_button = st.button("🚀 Run Benchmark Suite", type="primary", use_container_width=True)

    if bench_run_button:
        st.markdown("### 📈 Suite Results")

        async def execute_suite():
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

            if "Sweep" in bench_mode:
                settings = [
                    {"model": default_model, "temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
                    {"model": default_model, "temperature": 0.7, "top_p": b_top_p, "max_tokens": 128},
                    {"model": default_model, "temperature": 1.2, "top_p": 1.0, "max_tokens": 128},
                ]
            else:
                settings = [
                    {"model": default_model, "temperature": b_temp, "top_p": b_top_p, "max_tokens": 128}
                ]

            pace_delay = 0.0 if is_mock else 1.8

            with st.spinner(f"Running {len(settings)} settings × {b_repeats} repeats with rate-limit pacing..."):
                try:
                    runs = await run_benchmark_suite(
                        client=client,
                        prompts=[bench_prompt],
                        settings=settings,
                        repeats=int(b_repeats),
                        delay_between_runs_s=pace_delay,
                    )
                finally:
                    await client.aclose()
            return runs

        try:
            suite_runs = asyncio.run(execute_suite())
        except Exception as exc:
            st.error(f"Benchmark suite failed: {exc}")
        else:
            summary = summarize(suite_runs)
            groups = summary["groups"]

            # Summary Table
            table_rows = []
            for g in groups:
                table_rows.append(
                    {
                        "Temperature": f"{g['temperature']:.1f}",
                        "Top-P": f"{g['top_p']}" if g["top_p"] is not None else "default",
                        "Repeats": g["repeats"],
                        "Prompt Tok": f"{g['avg_prompt_tokens']:.0f}",
                        "Comp Tok": f"{g['avg_completion_tokens']:.0f}",
                        "TTFT (ms)": f"{g['avg_ttft_ms']:.1f}",
                        "Tokens / sec": f"{g['avg_tokens_per_second']:.1f}",
                        "Identical Outputs?": (
                            "✅ YES (Deterministic)"
                            if g["identical_across_repeats"]
                            else f"❌ NO ({g['distinct_outputs_count']} distinct variations)"
                        ),
                    }
                )

            st.dataframe(table_rows, use_container_width=True)

            # Output inspections
            st.markdown("#### 🔬 Detailed Output Text per Repeat (Nondeterminism Inspector)")
            for g in groups:
                label = (
                    f"🌡️ Temperature {g['temperature']:.1f} (Top-P: {g['top_p']}) — "
                    + ("Identical across repeats" if g["identical_across_repeats"] else f"{g['distinct_outputs_count']} variations across {g['repeats']} runs")
                )
                with st.expander(label, expanded=True):
                    for idx, out in enumerate(g["outputs"], 1):
                        st.markdown(f"**Run #{idx}:**")
                        st.code(out, language=None)

st.markdown("---")

# Pedagogical summary section
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
