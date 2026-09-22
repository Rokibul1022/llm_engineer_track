"""LLM Client — Live Service & Acceptance Demonstration Studio.

Provides two comprehensive demonstration environments mirroring static/index.html:
1. Tab 1: ⚡ Live Client, Speed Insights & Execution Pipeline
   - 5-Stage Execution Pipeline Stepper (Ingestion -> Dispatch -> Prefill -> Decode -> Billing)
   - Complete Speed Insights HUD (Tokens, Inference Time, Throughput, Cost Breakdown)
   - Visual Latency & Cost Decomposition (Prefill vs Decode Cards & Segmented Distribution)
   - Architecture & Mathematical Formulations Drawer
2. Tab 2: 🧪 Acceptance Test & Fault Injection Studio
   - 6 Reproducible Scenarios (500, 429, Timeout, 400 Fast-Fail, Malformed 200, Baseline)
   - Policy Decision HUD & Live Attempt Backoff Log

Run with: streamlit run streamlit_demo.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import httpx
import streamlit as st

from llm_client import AsyncLLMClient, ChatMessage, LLMRequest, Usage
from llm_client.schemas import AttemptLog
from scripts.run_benchmark import load_env_file

# Load credentials from .env
load_env_file(PROJECT_DIR / ".env")

SCENARIOS = {
    "🔁 500 Provider Error": {
        "script": ["500", "500", "ok"],
        "policy": "RETRY WITH EXPONENTIAL FULL JITTER",
        "action": "Retryable (5xx upstream server fault)",
        "note": "Transient backend server error; retries with full jitter backoff and cleanly recovers on attempt 3.",
    },
    "⏳ 429 Rate Limit": {
        "script": ["429", "429", "429"],
        "policy": "RETRY WITH EXPONENTIAL FULL JITTER",
        "action": "Retryable up to max_retries limit",
        "note": "Provider responds 429 Too Many Requests; client backs off exponentially until max_retries, then safely raises LLMRateLimitError.",
    },
    "⏱️ Timeout": {
        "script": ["timeout", "timeout", "timeout"],
        "policy": "RETRY WITH EXPONENTIAL FULL JITTER",
        "action": "Retryable up to max_retries limit",
        "note": "Connection dropped or socket timed out; safely retried with backoff before raising LLMTimeoutError.",
    },
    "🚫 400 Bad Request": {
        "script": ["400"],
        "policy": "FAST-FAIL CIRCUIT BREAKER",
        "action": "NON-RETRYABLE (Terminal client error)",
        "note": "Key acceptance criterion: 4xx client errors fast-fail immediately on Attempt 1 with ZERO retry quota waste.",
    },
    "🧩 Malformed Output (200 OK)": {
        "script": ["malformed"],
        "policy": "FAST-FAIL CIRCUIT BREAKER",
        "action": "NON-RETRYABLE (Data contract breach)",
        "note": "Provider returns HTTP 200 but JSON body violates Pydantic schema; terminates immediately as LLMMalformedResponseError.",
    },
    "✅ Baseline Success": {
        "script": ["ok"],
        "policy": "DIRECT EXECUTION",
        "action": "Normal execution",
        "note": "Healthy provider invocation completing on the first attempt without faults.",
    },
}

OUTCOME_ICONS = {
    "success": ("✅", "Success", "#10b981"),
    "rate_limited": ("⏳", "Rate limited (429)", "#f59e0b"),
    "provider_error": ("🔁", "Provider error (5xx)", "#f59e0b"),
    "timeout": ("⏱️", "Timeout", "#f59e0b"),
    "bad_request": ("🚫", "Bad request (4xx) — Fast-Fail", "#ef4444"),
    "malformed_response": ("🧩", "Malformed response — Fast-Fail", "#ef4444"),
}


def make_mock_transport(script: list[str]) -> httpx.MockTransport:
    remaining = list(script)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = remaining.pop(0)
        if outcome == "ok":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "This is a normal, successful response from the mock transport."}}],
                "usage": {"prompt_tokens": 14, "completion_tokens": 11, "total_tokens": 25},
            })
        if outcome == "500":
            return httpx.Response(500, text="internal server error")
        if outcome == "429":
            return httpx.Response(429, text="rate limit exceeded")
        if outcome == "400":
            return httpx.Response(400, text="invalid request: model not found")
        if outcome == "malformed":
            return httpx.Response(200, json={"unexpected": "schema_missing_choices"})
        if outcome == "timeout":
            raise httpx.TimeoutException("simulated socket timeout")
        raise ValueError(f"unknown scripted outcome: {outcome}")

    return httpx.MockTransport(handler)


async def run_fault_scenario(script: list[str], max_retries: int, base_delay_s: float, log_placeholder, result_placeholder):
    client = AsyncLLMClient(
        api_key="demo-key",
        max_retries=max_retries,
        base_delay_s=base_delay_s,
        max_delay_s=2.0,
        transport=make_mock_transport(script),
    )
    request = LLMRequest(model="gpt-demo", messages=[ChatMessage(role="user", content="hello")])
    rows: list[AttemptLog] = []

    def on_attempt(log: AttemptLog) -> None:
        rows.append(log)
        with log_placeholder.container():
            for r in rows:
                icon, label, color = OUTCOME_ICONS.get(r.outcome, ("ℹ️", r.outcome, "#94a3b8"))
                st.markdown(
                    f"""
                    <div style="background:#131720; border:1px solid #232b3b; border-left:4px solid {color}; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#e2e8f0;">{icon} Attempt {r.attempt} — {label}</span>
                            <span style="font-family:monospace; font-size:12px; color:#94a3b8; background:#0b0d13; padding:2px 8px; border-radius:4px;">{r.elapsed_ms:.0f} ms</span>
                        </div>
                        <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                            {'↳ <b>Will retry</b> after ' + f'{r.delay_before_retry_s:.2f}s backoff (exponential full jitter)' if r.will_retry else ('↳ 🛑 <b>Fast-Fail circuit breaker triggered</b> — terminated without retrying' if r.outcome != 'success' else '↳ 🎯 Resolved cleanly')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if r.detail and r.outcome != "success":
                    st.code(r.detail, language=None)

    try:
        response = await client.complete(request, on_attempt=on_attempt)
    except Exception as exc:
        with result_placeholder.container():
            st.error(f"🛑 Acceptance Verification: Safely raised `{type(exc).__name__}` as per circuit breaker policy.\n\n`{exc}`")
    else:
        with result_placeholder.container():
            st.success(
                f"✅ Client completed successfully after **{response.attempts} attempt(s)** in **{response.latency_ms:.0f} ms** total.\n\n"
                f"**Usage:** {response.usage.total_tokens} tokens ({response.usage.prompt_tokens} in / {response.usage.completion_tokens} out)\n\n"
                f"> {response.content}"
            )
    finally:
        await client.aclose()


def render_week1_full_studio(set_config: bool = False) -> None:
    if set_config:
        try:
            st.set_page_config(page_title="Week 1: Live Client & Acceptance Studio", layout="wide", page_icon="⚡")
        except Exception:
            pass

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; margin-bottom:12px;">
            <div>
                <h1 style="margin:0; font-size:24px; font-weight:800; letter-spacing:-0.5px;">⚡ Phase 1 Week 1 — Live Service & Acceptance Studio</h1>
                <p style="margin:4px 0 0; color:#94a3b8; font-size:14px;">
                    Production-grade Async LLM Client with real-time speed insights, latency decomposition, and fault-injection acceptance matrix.
                </p>
            </div>
            <div style="display:flex; gap:6px; margin-top:6px;">
                <span style="background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.3); font-size:11.5px; font-weight:700; padding:4px 8px; border-radius:6px;">HTTP POST /v1/stream</span>
                <span style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); font-size:11.5px; font-weight:700; padding:4px 8px; border-radius:6px;">Full Jitter Backoff</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_live, tab_acceptance = st.tabs([
        "⚡ Tab 1: Live Client, Speed Insights & Execution Pipeline",
        "🧪 Tab 2: Acceptance Test & Fault Injection Studio",
    ])

    # ==========================================================================
    # TAB 1: LIVE CLIENT, SPEED INSIGHTS & EXECUTION PIPELINE
    # ==========================================================================
    with tab_live:
        c1, c2 = st.columns([3, 1])
        with c1:
            prompt_input = st.text_area(
                "Prompt Input",
                value="Explain why Time to First Token (TTFT) differs from decode token latency in LLMs in 2 sentences.",
                height=90,
                key="w1_live_prompt_input",
                help="Input prompt streamed to openai/gpt-oss-120b using Server-Sent Events (SSE).",
            )
        with c2:
            model_id = st.text_input("Model ID", value=os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b"), disabled=True)
            temp_val = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="w1_live_temp_slider")

        stream_btn = st.button("🚀 Stream Live Completion", type="primary", use_container_width=True, key="w1_stream_btn")

        # Visual 5-Stage Execution Pipeline Stepper
        st.markdown("##### 📐 Backend Execution Pipeline")
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        step1_box = p_col1.empty()
        step2_box = p_col2.empty()
        step3_box = p_col3.empty()
        step4_box = p_col4.empty()
        step5_box = p_col5.empty()

        def draw_pipeline_steps(active_step: int, ttft_ms: float = 0, tok_count: int = 0, total_ms: float = 0):
            steps = [
                ("STEP 01", "CONTRACT", "📥 Request Ingestion", "Validates payload against Pydantic schema", "CONTRACT: VALID"),
                ("STEP 02", "TRANSPORT", "⚡ Async Dispatch", "Async HTTP POST /v1/stream with TLS", "SOCKET: CONNECTED"),
                ("STEP 03", "PHASE 1", "🚀 Prefill & TTFT", "Ingests prompt tokens and initializes KV cache", f"Prefill: {ttft_ms:.0f} ms" if ttft_ms > 0 else "Prefill: -- ms"),
                ("STEP 04", "PHASE 2", "✍️ Autoregressive Decode", "Iterative next-token loop streaming SSE chunks", f"Decode: {tok_count} tok" if tok_count > 0 else "Decode: -- tok"),
                ("STEP 05", "BILLING", "📦 Response & Billing", "Terminal usage parsed; decomposes latency & cost", f"Total: {total_ms:.0f} ms" if total_ms > 0 else "Total: -- ms"),
            ]
            placeholders = [step1_box, step2_box, step3_box, step4_box, step5_box]

            for idx, (num, tag, title, desc, stat) in enumerate(steps, 1):
                if idx < active_step:
                    border = "rgba(16,185,129,0.4)"
                    bg = "rgba(16,185,129,0.04)"
                    stat_color = "#34d399"
                    badge_color = "#34d399"
                elif idx == active_step:
                    border = "#3b82f6"
                    bg = "rgba(59,130,246,0.1)"
                    stat_color = "#93c5fd"
                    badge_color = "#60a5fa"
                else:
                    border = "#232b3b"
                    bg = "#131720"
                    stat_color = "#64748b"
                    badge_color = "#64748b"

                placeholders[idx - 1].markdown(
                    f"""
                    <div style="background:{bg}; border:1px solid {border}; border-radius:8px; padding:10px; min-height:115px; display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <div style="display:flex; justify-content:space-between; font-size:10px; font-weight:700; color:#64748b;">
                                <span>{num}</span>
                                <span style="color:{badge_color};">{tag}</span>
                            </div>
                            <div style="font-size:12px; font-weight:700; color:#e2e8f0; margin:4px 0 2px;">{title}</div>
                            <div style="font-size:10.5px; color:#94a3b8; line-height:1.25;">{desc}</div>
                        </div>
                        <div style="font-family:monospace; font-size:11px; font-weight:700; color:{stat_color}; background:#0b0d13; border:1px solid #1e2638; border-radius:4px; padding:3px 6px; margin-top:6px;">
                            {stat}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        draw_pipeline_steps(active_step=1)

        st.markdown("#### 📝 Streaming Output")
        out_placeholder = st.empty()
        insights_container = st.container()

        if stream_btn:
            draw_pipeline_steps(active_step=2)

            async def execute_live_stream():
                client = AsyncLLMClient(
                    api_key=os.getenv("LLM_API_KEY", "demo-key"),
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    max_retries=2,
                )
                req = LLMRequest(
                    model=model_id,
                    messages=[ChatMessage(role="user", content=prompt_input)],
                    temperature=temp_val,
                    max_tokens=256,
                    stream=True,
                )

                start_t = time.perf_counter()
                first_t: float | None = None
                collected_words = ""
                token_counter = 0
                final_usage: Usage | None = None

                def on_u(u: Usage):
                    nonlocal final_usage
                    final_usage = u

                try:
                    draw_pipeline_steps(active_step=3)

                    async for chunk in client.stream(req, on_usage=on_u):
                        now = time.perf_counter()
                        if first_t is None:
                            first_t = now
                            draw_pipeline_steps(active_step=4, ttft_ms=(first_t - start_t) * 1000, tok_count=1)

                        token_counter += 1
                        collected_words += chunk
                        out_placeholder.markdown(collected_words + " ▌")

                        if token_counter % 5 == 0:
                            draw_pipeline_steps(active_step=4, ttft_ms=(first_t - start_t) * 1000, tok_count=token_counter)

                    out_placeholder.markdown(collected_words)
                    end_t = time.perf_counter()

                    total_ms = (end_t - start_t) * 1000
                    ttft_ms = (first_t - start_t) * 1000 if first_t else total_ms
                    decode_ms = max(1.0, total_ms - ttft_ms)

                    comp_tokens = final_usage.completion_tokens if (final_usage and final_usage.completion_tokens > 0) else token_counter
                    prompt_tokens = final_usage.prompt_tokens if (final_usage and final_usage.prompt_tokens > 0) else len(prompt_input.split())
                    total_tokens = prompt_tokens + comp_tokens

                    decode_tps = comp_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0
                    overall_tps = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0
                    ms_per_tok = (decode_ms / comp_tokens) if comp_tokens > 0 else 0

                    prompt_cost = (prompt_tokens / 1_000_000) * 0.10
                    comp_cost = (comp_tokens / 1_000_000) * 0.40
                    total_cost = prompt_cost + comp_cost

                    draw_pipeline_steps(active_step=6, ttft_ms=ttft_ms, tok_count=comp_tokens, total_ms=total_ms)

                    # ==========================================================
                    # SPEED INSIGHTS PANELS (MIRRORING STATIC/INDEX.HTML)
                    # ==========================================================
                    with insights_container:
                        st.markdown("---")
                        st.markdown("### ⚡ Speed Insights HUD")

                        # Headline Bar
                        st.markdown(
                            f"""
                            <div style="background:#000; border-radius:10px; padding:12px 18px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; border:1px solid #232b3b;">
                                <span>Latency: <b style="color:#e2e8f0; font-size:16px;">{total_ms:.0f} ms</b></span>
                                <span><span style="color:#f59e0b;">⚡</span> Generation Speed: <b style="color:#38bdf8; font-size:16px;">{decode_tps:.1f} tok/s</b></span>
                                <span>💵 Est. Cost: <b style="color:#34d399; font-size:16px;">${total_cost:.6f}</b></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Row 1: Tokens Grid
                        st.markdown("**Tokens**")
                        t1, t2, t3 = st.columns(3)
                        t1.metric("Prompt Tokens", f"{prompt_tokens}")
                        t2.metric("Completion Tokens", f"{comp_tokens}")
                        t3.metric("Total Tokens", f"{total_tokens}")

                        # Row 2: Inference Time Grid
                        st.markdown("**Inference Time**")
                        tm1, tm2, tm3 = st.columns(3)
                        tm1.metric("Prefill (TTFT)", f"{ttft_ms:.0f} ms")
                        tm2.metric("Decode Time", f"{decode_ms:.0f} ms")
                        tm3.metric("Total Round-Trip", f"{total_ms:.0f} ms")

                        # Row 3: Throughput Grid
                        st.markdown("**Tokens / Second (Throughput)**")
                        tp1, tp2, tp3 = st.columns(3)
                        tp1.metric("Decode Throughput", f"{decode_tps:.1f} tok/s", delta="Generation speed")
                        tp2.metric("Overall Throughput", f"{overall_tps:.1f} tok/s")
                        tp3.metric("Time per Token", f"{ms_per_tok:.1f} ms/tok")

                        # Row 4: Cost Breakdown Grid
                        st.markdown("**Estimated API Cost** (Input: $0.10/1M · Output: $0.40/1M)")
                        c_col1, c_col2, c_col3 = st.columns(3)
                        c_col1.metric("Input Cost", f"${prompt_cost:.7f}")
                        c_col2.metric("Output Cost", f"${comp_cost:.7f}")
                        c_col3.metric("Total Cost", f"${total_cost:.6f}")

                        # ======================================================
                        # DETAILED LATENCY & COST DECOMPOSITION CARDS
                        # ======================================================
                        st.markdown("---")
                        st.markdown("### ⏱️ Latency & Cost Decomposition")

                        prefill_pct = min(100.0, max(0.0, (ttft_ms / total_ms) * 100))
                        decode_pct = 100.0 - prefill_pct

                        # Segmented Progress Bar
                        st.markdown(
                            f"""
                            <div style="height:26px; background:#0d1117; border-radius:8px; overflow:hidden; display:flex; margin:8px 0 16px; border:1px solid #1e2638;">
                                <div style="width:{prefill_pct}%; background:linear-gradient(90deg, #2563eb, #3b82f6); display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; color:#fff;">
                                    Prefill (TTFT): {ttft_ms:.0f} ms ({prefill_pct:.1f}%)
                                </div>
                                <div style="width:{decode_pct}%; background:linear-gradient(90deg, #059669, #10b981); display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; color:#fff;">
                                    Decode: {decode_ms:.0f} ms ({decode_pct:.1f}%)
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Dual Phase Cards
                        card_left, card_right = st.columns(2)
                        with card_left:
                            st.markdown(
                                f"""
                                <div style="background:#131720; border:1px solid #232b3b; border-left:4px solid #3b82f6; border-radius:10px; padding:16px;">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <div>
                                            <span style="font-size:10.5px; font-weight:700; color:#60a5fa;">PHASE 1</span>
                                            <div style="font-size:14px; font-weight:700; color:#e2e8f0;">Prefill & Time to First Token (TTFT)</div>
                                        </div>
                                        <span style="background:rgba(59,130,246,0.15); color:#60a5fa; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px;">{prefill_pct:.1f}%</span>
                                    </div>
                                    <div style="font-size:28px; font-weight:800; color:#93c5fd; font-family:monospace; margin:12px 0;">{ttft_ms:.0f} ms</div>
                                    <div style="display:flex; flex-direction:column; gap:6px; font-size:12px; color:#cbd5e1;">
                                        <div style="background:#0b0d13; padding:6px 10px; border-radius:6px;">📥 <b>Prompt Ingestion:</b> {prompt_tokens} tokens</div>
                                        <div style="background:#0b0d13; padding:6px 10px; border-radius:6px;">🌐 <b>Network transit & KV cache initialization</b></div>
                                        <div style="background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.25); color:#93c5fd; padding:6px 10px; border-radius:6px;">💰 <b>Input Cost:</b> ${prompt_cost:.7f} (@ $0.10/1M)</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with card_right:
                            st.markdown(
                                f"""
                                <div style="background:#131720; border:1px solid #232b3b; border-left:4px solid #10b981; border-radius:10px; padding:16px;">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <div>
                                            <span style="font-size:10.5px; font-weight:700; color:#34d399;">PHASE 2</span>
                                            <div style="font-size:14px; font-weight:700; color:#e2e8f0;">Autoregressive Generation (Decode)</div>
                                        </div>
                                        <span style="background:rgba(16,185,129,0.15); color:#34d399; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px;">{decode_pct:.1f}%</span>
                                    </div>
                                    <div style="font-size:28px; font-weight:800; color:#6ee7b7; font-family:monospace; margin:12px 0;">{decode_ms:.0f} ms</div>
                                    <div style="display:flex; flex-direction:column; gap:6px; font-size:12px; color:#cbd5e1;">
                                        <div style="background:#0b0d13; padding:6px 10px; border-radius:6px;">✍️ <b>Output Generated:</b> {comp_tokens} completion tokens</div>
                                        <div style="background:#0b0d13; padding:6px 10px; border-radius:6px;">🏎️ <b>Throughput:</b> {decode_tps:.1f} tok/s (~{ms_per_tok:.1f} ms/token)</div>
                                        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); color:#6ee7b7; padding:6px 10px; border-radius:6px;">💰 <b>Output Cost:</b> ${comp_cost:.7f} (@ $0.40/1M)</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Equation Footnote
                        st.markdown(
                            f"""
                            <div style="margin-top:14px; background:#080b10; border:1px solid #1e2638; border-radius:8px; padding:10px 14px; display:flex; justify-content:center; align-items:center; gap:8px; flex-wrap:wrap; font-family:monospace; font-size:13px;">
                                <span style="background:rgba(59,130,246,0.15); color:#93c5fd; padding:3px 8px; border-radius:4px;">🚀 Prefill: {ttft_ms:.0f} ms</span>
                                <span style="color:#64748b; font-weight:700;">+</span>
                                <span style="background:rgba(16,185,129,0.15); color:#6ee7b7; padding:3px 8px; border-radius:4px;">⚡ Decode: {decode_ms:.0f} ms</span>
                                <span style="color:#64748b; font-weight:700;">=</span>
                                <span style="background:rgba(245,158,11,0.15); color:#fcd34d; padding:3px 8px; border-radius:4px;">⏱️ Total: {total_ms:.0f} ms</span>
                                <span style="background:rgba(16,185,129,0.2); color:#34d399; padding:3px 8px; border-radius:4px; margin-left:8px;">💰 Est. Cost: ${total_cost:.6f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Architecture & Formulations Expander
                        with st.expander("📐 Technical Architecture Spec & Call Stack Drawer", expanded=False):
                            c_arch1, c_arch2 = st.columns(2)
                            with c_arch1:
                                st.markdown("**Python Call Stack**")
                                st.markdown(
                                    """
                                    - `1. Browser Client` ➔ `POST /v1/stream` with JSON prompt
                                    - `2. FastAPI Layer` ➔ validates `request: LLMRequest` via Pydantic
                                    - `3. AsyncLLMClient` ➔ `client.stream(request)` via `httpx.AsyncClient`
                                    - `4. Provider Inference` ➔ Prefill KV cache ➔ SSE streaming chunks
                                    - `5. Terminal Contract` ➔ `Usage(**chunk['usage'])` ➔ `LLMResponse`
                                    """
                                )
                            with c_arch2:
                                st.markdown("**Decomposition Formulations**")
                                st.markdown(
                                    """
                                    - **Total Latency:** $T_{\\text{total}} = \\text{TTFT}_{\\text{prefill}} + \\frac{N_{\\text{completion}}}{\\text{Throughput}}$
                                    - **Cost Billing:** $\\text{Cost} = (N_{\\text{in}} \\times \\frac{\\$0.10}{1\\text{M}}) + (N_{\\text{out}} \\times \\frac{\\$0.40}{1\\text{M}})$
                                    """
                                )

                except Exception as exc:
                    st.error(f"Streaming error: {exc}")
                finally:
                    await client.aclose()

            asyncio.run(execute_live_stream())

    # ==========================================================================
    # TAB 2: ACCEPTANCE TEST & FAULT INJECTION STUDIO
    # ==========================================================================
    with tab_acceptance:
        st.markdown(
            """
            <div style="font-size:13.5px; color:#94a3b8; margin-bottom:12px;">
                Simulate rate limits, malformed outputs, timeouts, and provider 5xx failures using an in-memory <code>httpx.MockTransport</code>.
                Demonstrates that the client handles each one predictably without uncontrolled retries.
            </div>
            """,
            unsafe_allow_html=True,
        )

        choice = st.selectbox("Select Failure / Acceptance Scenario", list(SCENARIOS.keys()), key="w1_tab2_scenario")
        scenario = SCENARIOS[choice]

        # Policy HUD
        hud_col1, hud_col2 = st.columns(2)
        with hud_col1:
            st.markdown(f"**Policy Action:** `{scenario['policy']}`")
        with hud_col2:
            st.markdown(f"**Classification:** `{scenario['action']}`")
        st.info(scenario["note"])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            max_retries = st.slider("max_retries", 1, 5, 3, key="w1_tab2_retries")
        with col_s2:
            base_delay = st.slider("base_delay_s", 0.05, 1.0, 0.2, step=0.05, key="w1_tab2_delay")

        run_fault = st.button("▶️ Run Acceptance Scenario", type="primary", use_container_width=True, key="w1_tab2_run_btn")

        if run_fault:
            st.markdown("#### 📋 Live Attempt Backoff Log")
            log_placeholder = st.empty()
            st.markdown("#### 🎯 Acceptance Resolution")
            result_placeholder = st.empty()

            with st.spinner("Executing scenario against mock transport..."):
                asyncio.run(
                    run_fault_scenario(
                        scenario["script"],
                        max_retries,
                        base_delay,
                        log_placeholder,
                        result_placeholder,
                    )
                )
        else:
            st.markdown("---")
            st.info("👆 Click **Run Acceptance Scenario** above to execute this failure simulation and watch live backoff logs.")

        st.divider()
        st.caption(
            "Retry policy: Only 429 and 5xx are retried using Exponential Backoff with Full Jitter. "
            "4xx Bad Request and 200 Malformed responses trigger instant circuit breaker termination on Attempt 1."
        )


def render_fault_injection_ui(set_config: bool = False) -> None:
    render_week1_full_studio(set_config=set_config)


if __name__ == "__main__":
    render_week1_full_studio(set_config=True)
