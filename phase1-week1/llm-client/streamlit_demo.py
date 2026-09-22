"""Fault-injection demo for the acceptance test:

    "Simulate rate limits, malformed outputs, timeouts, and provider
    failures. Demonstrate that the service handles them predictably
    without uncontrolled retries."

This does NOT call a real LLM provider. It wires AsyncLLMClient to an
httpx.MockTransport that plays back a scripted sequence of responses
(e.g. "fail twice with 500, then succeed"), so every scenario is
reproducible on demand instead of waiting to hit a real rate limit.

Run with: streamlit run streamlit_demo.py
"""
import asyncio
import time

import httpx
import streamlit as st

from llm_client import AsyncLLMClient, ChatMessage, LLMRequest
from llm_client.schemas import AttemptLog

SCENARIOS = {

    "✅ Success on first try": {
        "script": ["ok"],
        "note": "Baseline — no faults, one attempt.",
    },
    "🔁 Provider 500, then recovers": {
        "script": ["500", "500", "ok"],
        "note": "Two transient server errors, retried with backoff, then succeeds. "
                "This is the '5xx is retryable' behavior.",
    },
    "⏳ Rate limited (429), retries exhausted": {
        "script": ["429", "429", "429"],
        "note": "Provider keeps saying 'too many requests.' Retried up to max_retries, "
                "then raises LLMRateLimitError — it does not retry forever.",
    },
    "⏱️ Timeout, retries exhausted": {
        "script": ["timeout", "timeout", "timeout"],
        "note": "Connection never responds in time. Retried with backoff, "
                "then raises LLMTimeoutError.",
    },
    "🧩 Malformed 200 (bad schema)": {
        "script": ["malformed"],
        "note": "Provider returns HTTP 200 but the body doesn't match the expected shape. "
                "This is NOT retried — a malformed 200 usually repeats, so it fails "
                "immediately as LLMMalformedResponseError.",
    },
    "🚫 Bad request (400) — no uncontrolled retries": {
        "script": ["400"],
        "note": "This is the key acceptance criterion: a 4xx client error must fail on "
                "the FIRST attempt, never retried, because retrying an invalid request "
                "just burns your rate-limit budget for nothing.",
    },
}

def render_fault_injection_ui(set_config: bool = False) -> None:
    if set_config:
        try:
            st.set_page_config(page_title="LLM Client — Fault Injection Demo", layout="centered")
        except Exception:
            pass

    st.title("LLM Client — Fault Injection Demo")
    st.caption(
        "Simulates provider failures with a mock transport, no real API calls. "
        "Shows the client's retry policy handling each one predictably."
    )

    choice = st.selectbox("Scenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[choice]
    st.info(scenario["note"])

    col1, col2 = st.columns(2)
    max_retries = col1.slider("max_retries", 1, 5, 3)
    base_delay = col2.slider("base_delay_s", 0.05, 1.0, 0.2, step=0.05)

    run = st.button("Run scenario", type="primary")

    if run:
        st.markdown("#### 📋 Live Attempt Log")
        log_placeholder = st.empty()
        st.markdown("#### 🎯 Result")
        result_placeholder = st.empty()

        with st.spinner("Running against the mock transport..."):
            asyncio.run(
                run_scenario(
                    scenario["script"],
                    max_retries,
                    base_delay,
                    log_placeholder,
                    result_placeholder,
                )
            )
    else:
        st.markdown("---")
        st.info("👆 Click **Run scenario** above to simulate this failure mode and watch live retries.")

    st.divider()
    st.caption(
        "Retry policy: only 429 and 5xx are retried, with exponential backoff + full jitter. "
        "4xx and malformed-200 responses fail on the first attempt — see llm_client/client.py."
    )



def make_transport(script: list[str]) -> httpx.MockTransport:
    """Pops one scripted outcome per call. Raising past the end of the
    script means the test asked the client to make more attempts than
    the scenario has responses for — a bug in the demo, not the client."""
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


OUTCOME_LABEL = {
    "success": ("✅", "Success"),
    "rate_limited": ("⏳", "Rate limited (429)"),
    "provider_error": ("🔁", "Provider error (5xx)"),
    "timeout": ("⏱️", "Timeout"),
    "bad_request": ("🚫", "Bad request (4xx) — not retried"),
    "malformed_response": ("🧩", "Malformed response — not retried"),
}


async def run_scenario(script: list[str], max_retries: int, base_delay_s: float, log_placeholder, result_placeholder):
    client = AsyncLLMClient(
        api_key="demo-key",
        max_retries=max_retries,
        base_delay_s=base_delay_s,
        max_delay_s=2.0,
        transport=make_transport(script),
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
                    line += "  \n↳ 🛑 **Stopped — not retried (fail-fast policy)**"
                st.markdown(line)
                if r.detail and r.outcome != "success":
                    st.code(r.detail, language=None)

    try:
        response = await client.complete(request, on_attempt=on_attempt)
    except Exception as exc:  # noqa: BLE001 — demo surfaces whatever the client raises
        with result_placeholder.container():
            st.error(f"🛑 Client raised `{type(exc).__name__}`:\n\n{exc}")
    else:
        with result_placeholder.container():
            st.success(
                f"✅ Client completed successfully after **{response.attempts} attempt(s)** "
                f"in **{response.latency_ms:.0f} ms** total.\n\n"
                f"**Tokens:** {response.usage.total_tokens} total ({response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion)\n\n"
                f"> {response.content}"
            )
    finally:
        await client.aclose()


if __name__ == "__main__":
    render_fault_injection_ui(set_config=True)

