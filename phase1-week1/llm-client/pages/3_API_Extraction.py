"""Week 3 Module: Schema-Validated Extraction API Studio.

Built to match the exact visual style, stepper design, metric HUD, and layout of File 1 (Week 1 Studio).
Uses Groq openai/gpt-oss-120b with credentials loaded automatically from week1/week2 .env.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Ensure project paths are resolved
PROJECT_DIR = Path(__file__).resolve().parent.parent
WEEK3_DIR = PROJECT_DIR.parent.parent / "phase1-week3"

for p in [str(PROJECT_DIR), str(WEEK3_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
for env_file in [
    PROJECT_DIR / ".env",
    PROJECT_DIR.parent / ".env",
    WEEK3_DIR / ".env",
    Path.cwd() / ".env",
]:
    if env_file.exists():
        load_dotenv(env_file, override=False)

import streamlit as st

# Import extraction engine with hot-reload support
import importlib
import schemas
import prompts
import groq_client
import extractor

for _mod in [schemas, prompts, groq_client, extractor]:
    try:
        importlib.reload(_mod)
    except Exception:
        pass

from schemas import SCHEMA_REGISTRY
from prompts import PROMPTS, tool_def_for
from extractor import extract, semantic_checks
from groq_client import call_groq


def safe_html(content: str, target: Any = None) -> None:
    """Safely renders HTML without markdown CommonMark interference."""
    dedented = textwrap.dedent(content).strip()
    cleaned_lines = [
        line.lstrip() if line.startswith("    ") else line
        for line in dedented.splitlines()
    ]
    cleaned = "\n".join(cleaned_lines)
    if target is not None:
        if hasattr(target, "html"):
            target.html(cleaned)
        else:
            target.markdown(cleaned, unsafe_allow_html=True)
    else:
        if hasattr(st, "html"):
            st.html(cleaned)
        else:
            st.markdown(cleaned, unsafe_allow_html=True)


CUSTOM_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 50% -10%, #131c31 0%, #090d16 60%, #05070d 100%) !important;
    background-attachment: fixed !important;
}

code, kbd, pre, samp, [data-testid="stCodeBlock"] {
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] {
    background-color: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

[data-testid="stMetric"] {
    background: rgba(14, 20, 34, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.field-row {
    background: rgba(14, 20, 34, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.field-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #93c5fd;
}
.field-badge {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    font-size: 10.5px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 8px;
}
.field-val {
    font-size: 13px;
    color: #f1f5f9;
}
.warnbox-banner {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 10px;
    padding: 14px 18px;
    color: #fcd34d;
    font-size: 13px;
    margin-top: 14px;
}
</style>
"""

SAMPLE_PRESETS = {
    "ticket": {
        "Sample 1: Standard Bug Report (iOS 503)": "The iOS mobile app crashes immediately with error code 503 whenever a user taps the checkout button on the cart screen.",
        "Sample 2: Adversarial Negation (Not Billing)": "Customer says: this is NOT a billing issue, the app just crashes on login every time.",
        "Sample 3: Sarcastic Feedback (Adversarial Tone)": "Oh great, ANOTHER billing 'surprise' on my monthly credit card statement. Love it so much.",
        "Sample 4: Short Ambiguous Input (<20 chars)": "Crash on login",
        "Sample 5: Multi-Entity Feature Request": "We would love to connect our workspace with Slack, GitHub, and Jira so our engineering team at Acme Corp gets automatic pull request alerts.",
        "🔥 Error & Cleared: Overlong Summary (>200 chars Self-Correction)": "CRITICAL INCIDENT: Please provide an exhaustively long and detailed technical summary of this incident that is at least 450 characters long without abbreviating anything: The production PostgreSQL cluster experienced catastrophic disk block corruption on storage node db-01 at 04:12 UTC due to a faulty NVMe controller firmware panic, which immediately resulted in cascading split-brain condition across all secondary read-replicas in our Frankfurt primary availability zone.",
        "🔥 Error & Cleared: Forced Invalid Enum Recovery (urgency=SUPER_CRITICAL)": "The core authentication microservice has suffered total deadlock. Error code AUTH-9912. All login endpoints are unresponsive across production.",
        "🚫 Terminal Error: Unsupported Target Schema (Contract Rejection)": "Simulate contract breach with unregistered schema type 'unsupported_telemetry_schema' to test circuit breaker termination.",
    },
    "invoice": {
        "Sample 6: Standard Commercial Invoice ($1,240.50)": "Invoice #INV-2291 from Northwind Traders. Amount due: $1,240.50. Payment due by 2026-10-15.",
        "Sample 7: Missing Due Date (Fallback Rule)": "Invoice #INV-4099 issued by Apex Hosting Services. Total balance: 250.00 USD. No due date listed on the document.",
        "Sample 8: European VAT Invoice (1,890.00 EUR)": "Rechnung #DE-88219 von Berlin Cloud Services GmbH. Rechnungsbetrag: 1,890.00 EUR, zahlbar bis zum 2026-11-30.",
        "Sample 9: Missing / Ambiguous Amount (Net 30)": "Contractor billing memo #CM-104 from Global Design Studio. Terms: Net 30 days upon project milestone completion. Amount pending review.",
        "Sample 10: British Pounds Invoice (£3,750.25)": "Tax Invoice #UK-5501 from London Data Systems Ltd. Amount payable: £3,750.25 due on 2026-12-01.",
        "⚠️ Error Flagged: Negative Refund Memo & Missing Due Date": "CREDIT ADJUSTMENT MEMO #CR-9902: Refund issued to customer for account balance -450.00 USD due to billing system miscalculation. No payment due date is applicable.",
        "🔥 Error & Cleared: Forced Invalid Field Recovery (amount=STRING)": "Invoice #INV-8820 from Cyberdyne Systems. Amount: 1,500.00 USD. Payment due date: 2026-10-31.",
    }
}


def render_extraction_studio(set_config: bool = True) -> None:
    if set_config:
        try:
            st.set_page_config(
                page_title="API Extraction — Week 3 Studio",
                page_icon="🛡️",
                layout="wide",
            )
        except Exception:
            pass

    safe_html(CUSTOM_THEME_CSS)

    if "audit_history" not in st.session_state:
        st.session_state.audit_history = []

    # 1. Top Gradient Banner (Exact matching Week 1 header layout)
    safe_html(
        """
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(20, 30, 55, 0.65)); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); backdrop-filter: blur(12px); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 800; color: #38bdf8; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); padding: 3px 8px; border-radius: 6px; letter-spacing: 0.5px;">PHASE 1 · WEEK 3</span>
                    <span style="font-size: 11px; color: #64748b; font-family: monospace;">ExtractionService v1.0</span>
                </div>
                <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">Schema-Validated Extraction API Studio</h1>
                <p style="margin: 4px 0 0; color: #94a3b8; font-size: 13.5px; line-height: 1.4;">
                    Production extraction engine: tool calling on <code>openai/gpt-oss-120b</code>, Pydantic validation, self-correction retry, and semantic sanity checks.
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.3); color: #60a5fa; font-size: 11.5px; font-weight: 700; padding: 5px 10px; border-radius: 6px; display: inline-flex; align-items: center; gap: 6px;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: #60a5fa; display: inline-block;"></span> POST /extract
                </span>
                <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; font-size: 11.5px; font-weight: 700; padding: 5px 10px; border-radius: 6px; display: inline-flex; align-items: center; gap: 6px;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: #34d399; display: inline-block;"></span> SSE /extract/stream
                </span>
            </div>
        </div>
        """
    )

    # 2. Input Section & Interactive Controls (Columns [3, 1])
    c1, c2 = st.columns([3, 1])
    with c1:
        c_schema, c_preset = st.columns([1, 2])
        with c_schema:
            schema_type = st.selectbox(
                "Target Schema",
                options=["ticket", "invoice"],
                format_func=lambda s: "TicketExtraction (Tickets)" if s == "ticket" else "InvoiceExtraction (Invoices)",
                key="w3_studio_schema"
            )
        with c_preset:
            presets = SAMPLE_PRESETS[schema_type]
            preset_name = st.selectbox("Preset Test Examples (from prompt.md)", options=list(presets.keys()), key="w3_studio_preset")

        prompt_input = st.text_area(
            "Prompt Input Text",
            value=presets[preset_name],
            height=90,
            key="w3_studio_input",
            help="Raw unstructured text sent to the extraction engine."
        )

    with c2:
        st.text_input("Model ID", value="openai/gpt-oss-120b", disabled=True, key="w3_studio_model_id")
        exec_mode = st.radio("Execution Mode", ["🚀 Synchronous (/extract)", "⚡ Live SSE Stream (/stream)"], horizontal=True, key="w3_studio_mode")
        force_fault = st.checkbox(
            "🧪 Force Schema Violation on Att 1",
            value=("Forced" in preset_name),
            key="w3_studio_force_fault",
            help="Injects an invalid argument on Attempt 1 so the retry loop catches the ValidationError and the model self-corrects on Attempt 2, clearing the error."
        )

    run_btn = st.button("🚀 Run Structured Extraction", type="primary", use_container_width=True, key="w3_studio_run_btn")

    # 3. Visual 5-Stage Execution Pipeline Stepper (Exact matching Week 1 layout)
    st.markdown("##### 📐 Extraction Execution Pipeline")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    s1_box = p_col1.empty()
    s2_box = p_col2.empty()
    s3_box = p_col3.empty()
    s4_box = p_col4.empty()
    s5_box = p_col5.empty()

    def draw_pipeline_steps(active_step: int, attempt_count: int = 1, repair_msg: str = "", final_ms: float = 0, status_code: str = "READY"):
        steps = [
            ("STEP 01", "CONTRACT", "📥 Prompt Builder", "Generates system prompt & Pydantic tool spec", "STATUS: READY"),
            ("STEP 02", "INFERENCE", "⚡ Groq Tool Call", f"Forces tool_choice with gpt-oss-120b (att {attempt_count})", f"Attempt: {attempt_count}"),
            ("STEP 03", "VALIDATOR", "🛡️ Pydantic Validate", "Validates types, required fields, and enums", "VALIDATOR: PASS" if active_step >= 4 else "VALIDATOR: PENDING"),
            ("STEP 04", "REPAIR", "🔄 Retry Loop", "Feeds ValidationError back to assistant (max 2)", repair_msg if repair_msg else ("PASSED ATTEMPT 1" if active_step >= 5 else "BUDGET: 2 RETRIES")),
            ("STEP 05", "SANITY", "📦 Output & Flags", "Extracts clean payload & checks negation/sarcasm", f"{final_ms:.0f} ms · {status_code}" if final_ms > 0 else "STATUS: PENDING"),
        ]
        placeholders = [s1_box, s2_box, s3_box, s4_box, s5_box]

        for idx, (num, tag, title, desc, stat) in enumerate(steps, 1):
            if idx < active_step:
                border = "rgba(16,185,129,0.4)"
                bg = "rgba(16,185,129,0.06)"
                stat_color = "#34d399"
                badge_color = "#34d399"
            elif idx == active_step:
                border = "#3b82f6"
                bg = "rgba(59,130,246,0.12)"
                stat_color = "#93c5fd"
                badge_color = "#60a5fa"
            else:
                border = "#1e293b"
                bg = "#0f141f"
                stat_color = "#64748b"
                badge_color = "#64748b"

            card = (
                f'<div style="background:{bg}; border:1px solid {border}; border-radius:10px; padding:12px; min-height:120px; display:flex; flex-direction:column; justify-content:space-between;">'
                f'<div>'
                f'<div style="display:flex; justify-content:space-between; font-size:10px; font-weight:700; color:#64748b;">'
                f'<span>{num}</span>'
                f'<span style="color:{badge_color};">{tag}</span>'
                f'</div>'
                f'<div style="font-size:12.5px; font-weight:700; color:#e2e8f0; margin:4px 0 2px;">{title}</div>'
                f'<div style="font-size:10.5px; color:#94a3b8; line-height:1.3;">{desc}</div>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\', monospace; font-size:11px; font-weight:700; color:{stat_color}; background:#080b12; border:1px solid #1a2336; border-radius:5px; padding:4px 6px; margin-top:6px; text-align:center;">'
                f'{stat}'
                f'</div>'
                f'</div>'
            )
            safe_html(card, target=placeholders[idx - 1])

    draw_pipeline_steps(active_step=1)

    # Output & Insights Containers
    st.markdown("#### 📝 Extraction Output & Insights")
    insights_container = st.container()

    if run_btn:
        draw_pipeline_steps(active_step=2)
        t_start = time.perf_counter()

        if "Synchronous" in exec_mode:
            try:
                draw_pipeline_steps(active_step=3)
                target_schema = "unsupported_telemetry_schema" if "Unsupported Target Schema" in preset_name else schema_type
                res = extract(target_schema, prompt_input, force_initial_fault=force_fault)
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                attempts = res["attempts"]
                status = res["status"]
                data = res["data"]
                flags = semantic_checks(data, prompt_input)
                attempt_errors = res.get("attempt_errors", [])

                draw_pipeline_steps(
                    active_step=6,
                    attempt_count=attempts,
                    repair_msg=f"REPAIRED & CLEARED (ATT {attempts})" if attempts > 1 else "NO RETRIES NEEDED",
                    final_ms=elapsed_ms,
                    status_code="200 OK"
                )

                with insights_container:
                    # Metric HUD (Columns 4)
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Latency", f"{elapsed_ms:.0f} ms", delta="Groq API Roundtrip")
                    with m2:
                        st.metric("Attempts Used", f"{attempts} / 3", delta="Repaired & Cleared" if attempts > 1 else "Direct Pass")
                    with m3:
                        st.metric("Schema Compliance", "100% VALID", delta=f"{SCHEMA_REGISTRY[schema_type].__name__}")
                    with m4:
                        st.metric("Semantic Sanity", f"{len(flags)} Flags", delta="Caution" if flags else "Clean")

                    # Self-Correction Recovery Audit Card (Shown if an error was caught and cleared)
                    if attempts > 1 or attempt_errors:
                        err_items = "<br>".join(f"<code style='color:#fca5a5;'>• {e}</code>" for e in attempt_errors)
                        safe_html(
                            f'<div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-left: 4px solid #ef4444; border-radius: 8px; padding: 12px 16px; margin: 12px 0 16px; font-size: 13px;">'
                            f'<div style="font-weight: 700; color: #ef4444; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">'
                            f'<span>❌ Attempt 1: Schema ValidationError Caught</span>'
                            f'<span style="background: rgba(239, 68, 68, 0.2); color: #fca5a5; font-size: 11px; padding: 2px 6px; border-radius: 4px;">Intercepted by Retry Loop</span>'
                            f'</div>'
                            f'{err_items}'
                            f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(239, 68, 68, 0.2); font-weight: 700; color: #34d399; display: flex; align-items: center; gap: 6px;">'
                            f'<span>✅ Attempt {attempts}: Error Cleared!</span>'
                            f'<span style="color: #94a3b8; font-weight: 400; font-size: 12px;">Validation error fed back to assistant; model self-corrected arguments.</span>'
                            f'</div>'
                            f'</div>'
                        )

                    # Decomposition Cards (2 Columns)
                    col_out, col_spec = st.columns([1.2, 1])
                    with col_out:
                        st.markdown("##### 📋 Validated Structured Output")
                        for k, v in data.items():
                            val_str = json.dumps(v) if isinstance(v, list) else str(v)
                            card_field = (
                                f'<div class="field-row">'
                                f'<span class="field-key">{k}</span>'
                                f'<div><span class="field-badge">VALID</span><span class="field-val">{val_str}</span></div>'
                                f'</div>'
                            )
                            safe_html(card_field)

                        if flags:
                            flags_html = "<br>".join(f"• {f}" for f in flags)
                            safe_html(
                                f'<div class="warnbox-banner">'
                                f'<strong>⚠️ Semantic Warnings (schema passed, content questionable):</strong><br>{flags_html}'
                                f'</div>'
                            )

                    with col_spec:
                        st.markdown("##### 📐 Active Schema Specification")
                        st.json(SCHEMA_REGISTRY[schema_type].model_json_schema())

                # Record in session state
                st.session_state.audit_history.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "schema": SCHEMA_REGISTRY[schema_type].__name__,
                    "attempts": attempts,
                    "status": status,
                    "mode": "sync (/extract)",
                    "latency": f"{elapsed_ms:.0f} ms",
                    "flags": f"{len(flags)} flag(s)" if flags else "none",
                })

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                draw_pipeline_steps(active_step=4, attempt_count=3, repair_msg="BUDGET EXHAUSTED", final_ms=elapsed_ms, status_code="FAILED")
                with insights_container:
                    st.error(f"🛑 Extraction Failure: {exc}")
                st.session_state.audit_history.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "schema": SCHEMA_REGISTRY[schema_type].__name__,
                    "attempts": 3,
                    "status": "failed",
                    "mode": "sync (/extract)",
                    "latency": f"{elapsed_ms:.0f} ms",
                    "flags": "none",
                })

        else:
            # STREAMING SSE MODE
            model_cls = SCHEMA_REGISTRY[schema_type]
            tool_def = tool_def_for(schema_type, model_cls)
            tool_name = f"extract_{schema_type}_data"
            messages = [
                {"role": "system", "content": PROMPTS[schema_type]},
                {"role": "user", "content": prompt_input},
            ]

            stream_area = st.empty()
            accumulated = ""

            try:
                draw_pipeline_steps(active_step=3)
                stream = call_groq(messages, tool_def, tool_name=tool_name, stream=True)
                tok_count = 0

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        arg_piece = delta.tool_calls[0].function.arguments or ""
                        accumulated += arg_piece
                        tok_count += 1
                        stream_area.code(accumulated, language="json")
                        draw_pipeline_steps(active_step=4, attempt_count=1, repair_msg=f"STREAMING ({tok_count} deltas)")

                # Final validation
                parsed = model_cls.model_validate(json.loads(accumulated))
                data = parsed.model_dump()
                flags = semantic_checks(data, prompt_input)
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                draw_pipeline_steps(active_step=6, attempt_count=1, repair_msg="SINGLE PASS (STREAM)", final_ms=elapsed_ms, status_code="200 OK")

                with insights_container:
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Latency", f"{elapsed_ms:.0f} ms", delta="Streaming SSE")
                    with m2:
                        st.metric("Attempts Used", "1 / 1", delta="Single-Pass Stream")
                    with m3:
                        st.metric("Schema Compliance", "100% VALID", delta=f"{model_cls.__name__}")
                    with m4:
                        st.metric("Semantic Sanity", f"{len(flags)} Flags", delta="Caution" if flags else "Clean")

                    col_out, col_spec = st.columns([1.2, 1])
                    with col_out:
                        st.markdown("##### 📋 Validated Structured Output")
                        for k, v in data.items():
                            val_str = json.dumps(v) if isinstance(v, list) else str(v)
                            card_field = (
                                f'<div class="field-row">'
                                f'<span class="field-key">{k}</span>'
                                f'<div><span class="field-badge" style="background:rgba(59,130,246,0.18); color:#60a5fa; border-color:rgba(59,130,246,0.35);">STREAM VALID</span><span class="field-val">{val_str}</span></div>'
                                f'</div>'
                            )
                            safe_html(card_field)

                        if flags:
                            flags_html = "<br>".join(f"• {f}" for f in flags)
                            safe_html(
                                f'<div class="warnbox-banner">'
                                f'<strong>⚠️ Semantic Warnings:</strong><br>{flags_html}'
                                f'</div>'
                            )

                    with col_spec:
                        st.markdown("##### 📐 Raw Streamed Arguments Buffer")
                        st.code(accumulated, language="json")

                st.session_state.audit_history.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "schema": model_cls.__name__,
                    "attempts": 1,
                    "status": "success",
                    "mode": "stream (/extract/stream)",
                    "latency": f"{elapsed_ms:.0f} ms",
                    "flags": f"{len(flags)} flag(s)" if flags else "none",
                })

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                draw_pipeline_steps(active_step=4, attempt_count=1, repair_msg="STREAM ERROR", final_ms=elapsed_ms, status_code="FAILED")
                with insights_container:
                    st.error(f"🛑 Streaming Validation Failure: {exc}")
                st.session_state.audit_history.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "schema": model_cls.__name__,
                    "attempts": 1,
                    "status": "failed",
                    "mode": "stream (/extract/stream)",
                    "latency": f"{elapsed_ms:.0f} ms",
                    "flags": "none",
                })

    # 4. Architecture & Formulations Expander (Exact matching Week 1 drawer layout)
    with st.expander("📐 Technical Architecture Spec & Call Stack Drawer", expanded=False):
        c_arch1, c_arch2 = st.columns(2)
        with c_arch1:
            st.markdown("**Python Call Stack & Repair Flow**")
            st.markdown(
                """
                - `1. Browser Client` ➔ `POST /extract` or `POST /extract/stream`
                - `2. FastAPI Routing` ➔ unpacks `{schema_name, text}` via Pydantic
                - `3. ExtractionService` ➔ selects prompt from `PROMPTS[schema]` & builds JSON Schema tool
                - `4. Groq Client` ➔ executes completion with forced `tool_choice` on `openai/gpt-oss-120b`
                - `5. Self-Correction Loop` ➔ feeds `ValidationError` back as `tool` role message (max 2 retries)
                - `6. Semantic Sanity Pass` ➔ runs heuristic spot-rules (negation, sarcasm, summary grounding)
                """
            )
        with c_arch2:
            st.markdown("**Guaranteed Architectural Invariants**")
            st.markdown(
                """
                - **Strict Typed Failure:** Unknown schema raises typed `ExtractionFailure`, never raw `KeyError`.
                - **Single-Pass Streaming:** `/extract/stream` streams argument deltas live and validates once at the end without mid-stream retry.
                - **Proxy Pre-Validation Interception:** Groq's 400 `tool_use_failed` errors with `failed_generation` are caught and automatically channeled into the retry loop.
                """
            )

    # 5. Session Audit Trail Table
    st.markdown("---")
    st.markdown("### 📜 Session Audit Trail")
    if st.session_state.audit_history:
        st.dataframe(st.session_state.audit_history, use_container_width=True)
    else:
        st.caption("No extractions run yet in this session. Click **Run Structured Extraction** above.")


if __name__ == "__main__":
    render_extraction_studio(set_config=True)
