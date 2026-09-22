"""LLM Client — Live Service & Acceptance Demonstration Studio.

Provides two comprehensive demonstration environments:
1. Tab 1: ⚡ Live Client, Speed Insights & Latency Decomposition
2. Tab 2: 🧪 Acceptance Test & Fault Injection Studio

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
    "✅ Success on first try": {
        "script": ["ok"],
        "note": "Baseline — no faults, one attempt.",
    },
    "🔁 Provider 500, then recovers": {
        "script": ["500", "500", "ok"],
        "note": "Two transient server errors, retried with backoff, then succeeds. "
                "This demonstrates that 5xx server errors are retryable.",
    },
    "⏳ Rate limited (429), retries exhausted": {
        "script": ["429", "429", "429"],
        "note": "Provider keeps returning 429 Too Many Requests. Retried up to max_retries with jitter, "
                "then safely raises LLMRateLimitError — never loops indefinitely.",
    },
    "⏱️ Timeout, retries exhausted": {
        "script": ["timeout", "timeout", "timeout"],
        "note": "Connection never responds in time. Retried with exponential backoff, "
                "then raises LLMTimeoutError.",
    },
    "🧩 Malformed 200 (bad schema)": {
        "script": ["malformed"],
        "note": "Provider returns HTTP 200 but body doesn't match expected schema. "
                "Fails immediately on Attempt 1 as LLMMalformedResponseError (circuit breaker).",
    },
    "🚫 Bad request (400) — no uncontrolled retries": {
        "script": ["400"],
        "note": "Key acceptance criterion: 4xx client errors fast-fail on Attempt 1 "
                "without wasting retry quota.",
    },
}

OUTCOME_LABEL = {
    "success": ("✅", "Success"),
    "rate_limited": ("⏳", "Rate limited (429)"),
    "provider_error": ("🔁", "Provider error (5xx)"),
    "timeout": ("⏱️", "Timeout"),
    "bad_request": ("🚫", "Bad request (4xx) — fast fail"),
    "malformed_response": ("🧩", "Malformed response — fast fail"),
}


def make_mock_transport(script: list[str]) -> httpx.MockTransport:
    remaining = list(script)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = remaining.pop(0)
        if outcome == "ok":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "This is a normal, successful response."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            })
        if outcome == "500":
            return httpx.Response(500, text="internal server error")
        if outcome == "429":
            return httpx.Response(429, text="rate limit exceeded")
        if outcome == "400":
            return httpx.Response(400, text="invalid request: unknown model")
        if outcome == "malformed":
            return httpx.Response(200, json={"unexpected": "shape, no choices/usage"})
        if outcome == "timeout":
            raise httpx.TimeoutException("simulated timeout")
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
                icon, label = OUTCOME_LABEL.get(r.outcome, ("ℹ️", r.outcome))
                line = f"{icon} **Attempt {r.attempt}** — {label} · `{r.elapsed_ms:.0f} ms` elapsed"
                if r.will_retry:
                    line += f"  \n↳ *Will retry after {r.delay_before_retry_s:.2f}s backoff (full jitter)*"
                elif r.outcome != "success":
                    line += "  \n↳ 🛑 **Fast-Fail circuit breaker activated (no retry)**"
                st.markdown(line)
                if r.detail and r.outcome != "success":
                    st.code(r.detail, language=None)

    try:
        response = await client.complete(request, on_attempt=on_attempt)
    except Exception as exc:
        with result_placeholder.container():
            st.error(f"🛑 Client safely raised `{type(exc).__name__}`:\n\n{exc}")
    else:
        with result_placeholder.container():
            st.success(
                f"✅ Client completed successfully after **{response.attempts} attempt(s)** "
                f"in **{response.latency_ms:.0f} ms** total.\n\n"
                f"**Tokens:** {response.usage.total_tokens} total ({response.usage.prompt_tokens} in / {response.usage.completion_tokens} out)\n\n"
                f"> {response.content}"
            )
    finally:
        await client.aclose()


def render_week1_full_studio(set_config: bool = False) -> None:
    if set_config:
        try:
            st.set_page_config(page_title="Week 1: Live Client & Acceptance Studio", layout="wide", page_icon="🛡️")
        except Exception:
            pass

    st.title("🛡️ Phase 1 Week 1 — Live Service & Acceptance Test Studio")
    st.caption(
        "Demonstrates both the **Live Resilient Client with Speed Insights** and the "
        "**Acceptance Test & Fault-Injection Engine**."
    )

    tab_live, tab_acceptance = st.tabs([
        "⚡ Tab 1: Live Client, Speed Insights & Latency Decomposition",
        "🧪 Tab 2: Acceptance Test & Fault Injection Studio",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: LIVE CLIENT & SPEED INSIGHTS
    # --------------------------------------------------------------------------
    with tab_live:
        st.subheader("⚡ Live Client & Real-Time Speed Insights")
        st.caption("Call the live model (`openai/gpt-oss-120b`), stream tokens, and see real-time TTFT vs decode latency decomposition.")

        c1, c2 = st.columns([3, 1])
        with c1:
            prompt_text = st.text_area(
                "Prompt Input",
                value="Explain the difference between Time-to-First-Token (TTFT) and decode latency in 2 sentences.",
                height=95,
                key="w1_live_prompt",
            )
        with c2:
            model_id = st.text_input("Model ID", value=os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b"), disabled=True)
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="w1_live_temp")

        btn_stream = st.button("🚀 Stream Live Completion", type="primary", key="w1_live_btn")

        if btn_stream:
            metrics_box = st.empty()
            st.markdown("#### 📝 Streaming Output:")
            out_box = st.empty()

            async def do_live_stream():
                client = AsyncLLMClient(
                    api_key=os.getenv("LLM_API_KEY", "demo-key"),
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    max_retries=2,
                )
                req = LLMRequest(
                    model=model_id,
                    messages=[ChatMessage(role="user", content=prompt_text)],
                    temperature=temperature,
                    max_tokens=256,
                    stream=True,
                )

                start_time = time.perf_counter()
                first_token_time: float | None = None
                tokens_accumulated = ""
                tok_count = 0
                usage_record: Usage | None = None

                def record_u(u: Usage):
                    nonlocal usage_record
                    usage_record = u

                try:
                    async for token in client.stream(req, on_usage=record_u):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        tok_count += 1
                        tokens_accumulated += token
                        out_box.markdown(tokens_accumulated + " ▌")

                        cur_ttft = (first_token_time - start_time) * 1000 if first_token_time else (now - start_time) * 1000
                        cur_tps = (tok_count / (now - first_token_time)) if (first_token_time and now > first_token_time) else 0

                        with metrics_box.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("⏱️ TTFT (Prefill)", f"{cur_ttft:.0f} ms")
                            m2.metric("⚡ Decode Speed", f"{cur_tps:.1f} tok/s")
                            m3.metric("🔢 Tokens", f"{tok_count}")
                            m4.metric("⏱️ Elapsed", f"{now - start_time:.2f} s")

                    out_box.markdown(tokens_accumulated)
                    end_time = time.perf_counter()

                    total_dur_ms = (end_time - start_time) * 1000
                    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_dur_ms
                    decode_ms = max(1.0, total_dur_ms - ttft_ms)

                    c_toks = usage_record.completion_tokens if (usage_record and usage_record.completion_tokens > 0) else tok_count
                    p_toks = usage_record.prompt_tokens if (usage_record and usage_record.prompt_tokens > 0) else len(prompt_text.split())
                    final_tps = (c_toks / (decode_ms / 1000.0))

                    # Cost model: $0.10 / 1M prompt, $0.40 / 1M completion
                    prompt_cost = (p_toks / 1_000_000) * 0.10
                    comp_cost = (c_toks / 1_000_000) * 0.40
                    total_cost = prompt_cost + comp_cost

                    with metrics_box.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("⏱️ TTFT (Prefill)", f"{ttft_ms:.0f} ms")
                        m2.metric("⚡ Decode Speed", f"{final_tps:.1f} tok/s")
                        m3.metric("🔢 Total Tokens", f"{p_toks + c_toks}", f"{p_toks} prompt / {c_toks} completion")
                        m4.metric("💰 Est. Cost", f"${total_cost:.6f}")

                    st.markdown("##### 📊 Physical Latency Decomposition")
                    prefill_pct = min(100.0, max(0.0, (ttft_ms / total_dur_ms) * 100))
                    decode_pct = 100.0 - prefill_pct

                    col_bar1, col_bar2 = st.columns([int(prefill_pct) or 1, int(decode_pct) or 1])
                    with col_bar1:
                        st.info(f"**Phase 1: Prefill / TTFT**\n\n`{ttft_ms:.0f} ms` ({prefill_pct:.1f}%)")
                    with col_bar2:
                        st.success(f"**Phase 2: Autoregressive Decode**\n\n`{decode_ms:.0f} ms` ({decode_pct:.1f}%)")

                except Exception as exc:
                    st.error(f"Live execution error: {exc}")
                finally:
                    await client.aclose()

            asyncio.run(do_live_stream())

    # --------------------------------------------------------------------------
    # TAB 2: ACCEPTANCE TEST & FAULT INJECTION STUDIO
    # --------------------------------------------------------------------------
    with tab_acceptance:
        st.subheader("🧪 Acceptance Test & Fault Injection Studio")
        st.caption(
            "Wires `AsyncLLMClient` to a scripted `httpx.MockTransport`. "
            "Simulates rate limits, transient 5xx errors, timeouts, and 4xx bad requests without burning live API credits."
        )

        choice = st.selectbox("Failure / Acceptance Scenario", list(SCENARIOS.keys()), key="w1_scenario_choice")
        scenario = SCENARIOS[choice]
        st.info(scenario["note"])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            max_retries = st.slider("max_retries", 1, 5, 3, key="w1_max_retries")
        with col_s2:
            base_delay = st.slider("base_delay_s", 0.05, 1.0, 0.2, step=0.05, key="w1_base_delay")

        run_fault = st.button("▶️ Run Acceptance Scenario", type="primary", key="w1_run_fault_btn")

        if run_fault:
            st.markdown("#### 📋 Live Attempt Log")
            log_placeholder = st.empty()
            st.markdown("#### 🎯 Acceptance Verdict")
            result_placeholder = st.empty()

            with st.spinner("Simulating scenario against mock transport..."):
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
            st.info("👆 Click **Run Acceptance Scenario** to simulate this failure mode and watch backoff logs.")

        st.divider()
        st.caption(
            "Retry policy: Only 429 and 5xx are retried using Exponential Backoff with Full Jitter. "
            "4xx Bad Request and 200 Malformed responses trigger instant circuit breaker termination."
        )


def render_fault_injection_ui(set_config: bool = False) -> None:
    render_week1_full_studio(set_config=set_config)


if __name__ == "__main__":
    render_week1_full_studio(set_config=True)
