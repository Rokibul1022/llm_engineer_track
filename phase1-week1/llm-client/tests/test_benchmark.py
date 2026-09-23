"""Unit tests for the benchmarking module using respx-mocked HTTP streams."""
from __future__ import annotations

import httpx
import pytest
import respx

from llm_client import (
    AsyncLLMClient,
    BenchmarkRun,
    ChatMessage,
    LLMRequest,
    run_benchmark_suite,
    run_single_benchmark,
    summarize,
)

URL = "https://api.example.com/v1/chat/completions"


@pytest.fixture
def client():
    return AsyncLLMClient(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        max_retries=2,
        base_delay_s=0.01,
        max_delay_s=0.02,
    )


@pytest.mark.asyncio
@respx.mock
async def test_run_single_benchmark_computes_ttft_and_tps(client):
    sse_data = (
        'data: {"choices": [{"delta": {"content": "The"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " capital"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " is"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " Paris."}}]}\n\n'
        'data: {"usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}\n\n'
        'data: [DONE]\n\n'
    )
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            text=sse_data,
            headers={"content-type": "text/event-stream"},
        )
    )

    req = LLMRequest(
        model="openai/gpt-oss-120b",
        messages=[ChatMessage(role="user", content="What is the capital of France?")],
        temperature=0.0,
        top_p=1.0,
    )

    run = await run_single_benchmark(client, req)

    assert run.prompt == "What is the capital of France?"
    assert run.output_text == "The capital is Paris."
    assert run.completion_tokens == 4
    assert run.prompt_tokens == 8
    assert run.ttft_s >= 0.0
    assert run.total_s >= run.ttft_s
    assert run.tokens_per_second > 0.0
    assert len(run.token_timings) == 4
    assert run.token_timings[0].token_index == 1


def test_summarize_flags_nondeterminism():
    # Construct synthetic runs at temp=0.0 (identical) and temp=1.2 (divergent)
    deterministic_runs = [
        BenchmarkRun(
            prompt="Explain gravity",
            model="openai/gpt-oss-120b",
            temperature=0.0,
            top_p=1.0,
            prompt_tokens=10,
            completion_tokens=20,
            ttft_s=0.08,
            total_s=0.30,
            tokens_per_second=66.6,
            output_text="Gravity is a fundamental force of nature that attracts masses.",
        )
        for _ in range(3)
    ]

    divergent_runs = [
        BenchmarkRun(
            prompt="Explain gravity",
            model="openai/gpt-oss-120b",
            temperature=1.2,
            top_p=1.0,
            prompt_tokens=10,
            completion_tokens=22,
            ttft_s=0.09,
            total_s=0.32,
            tokens_per_second=68.7,
            output_text=f"Gravity is an exotic curvature variation {i} in spacetime!",
        )
        for i in range(3)
    ]

    all_runs = deterministic_runs + divergent_runs
    summary = summarize(all_runs)

    assert summary["total_runs"] == 6
    groups = summary["groups"]
    assert len(groups) == 2

    # Group 1: temp 0.0
    g_det = next(g for g in groups if g["temperature"] == 0.0)
    assert g_det["identical_across_repeats"] is True
    assert g_det["distinct_outputs_count"] == 1
    assert g_det["repeats"] == 3

    # Group 2: temp 1.2
    g_div = next(g for g in groups if g["temperature"] == 1.2)
    assert g_div["identical_across_repeats"] is False
    assert g_div["distinct_outputs_count"] == 3
    assert g_div["repeats"] == 3


@pytest.mark.asyncio
@respx.mock
async def test_run_benchmark_suite_cartesian_product(client):
    sse_data = (
        'data: {"choices": [{"delta": {"content": "Sample output."}}]}\n\n'
        'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}\n\n'
        'data: [DONE]\n\n'
    )
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            text=sse_data,
            headers={"content-type": "text/event-stream"},
        )
    )

    prompts = ["Short prompt", "Another short prompt"]
    settings = [
        {"temperature": 0.0, "top_p": 1.0},
        {"temperature": 0.7, "top_p": 0.9},
    ]
    repeats = 2

    # Cartesian product: 2 prompts * 2 settings * 2 repeats = 8 runs
    runs = await run_benchmark_suite(client, prompts, settings, repeats=repeats)

    assert len(runs) == 8
    for r in runs:
        assert r.output_text == "Sample output."
        assert r.completion_tokens == 2
