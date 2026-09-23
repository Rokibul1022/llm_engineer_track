"""Server-side, scriptable token generation benchmarking module.

Measures:
- Time to First Token (TTFT) using monotonic clock time.perf_counter().
- Overall tokens/sec and decode throughput.
- Output determinism across temperature/top_p parameter sweeps.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .schemas import ChatMessage, LLMRequest, Usage

if TYPE_CHECKING:
    from .client import AsyncLLMClient


class TokenTimingSample(BaseModel):
    token_index: int
    elapsed_since_start_s: float


class BenchmarkRun(BaseModel):
    prompt: str
    model: str
    temperature: float
    top_p: float | None = None
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    tokens_per_second: float
    output_text: str
    token_timings: list[TokenTimingSample] = Field(default_factory=list)


async def run_single_benchmark(
    client: AsyncLLMClient,
    request: LLMRequest,
) -> BenchmarkRun:
    """Stream one completion, timing first-token latency (TTFT) and
    overall tokens/sec using client.stream().

    Reuses LLMRequest and Usage schemas from schemas.py.
    """
    prompt_text = ""
    for msg in request.messages:
        if msg.role == "user":
            prompt_text = msg.content
            break
    if not prompt_text and request.messages:
        prompt_text = request.messages[-1].content

    stream_request = request.model_copy(update={"stream": True})

    collected_chunks: list[str] = []
    token_timings: list[TokenTimingSample] = []
    recorded_usage: Usage | None = None

    def capture_usage(u: Usage) -> None:
        nonlocal recorded_usage
        recorded_usage = u

    start_time = time.perf_counter()
    first_token_time: float | None = None

    token_idx = 0
    async for chunk in client.stream(stream_request, on_usage=capture_usage):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_idx += 1
        collected_chunks.append(chunk)
        token_timings.append(
            TokenTimingSample(
                token_index=token_idx,
                elapsed_since_start_s=round(now - start_time, 6),
            )
        )

    end_time = time.perf_counter()
    total_s = end_time - start_time

    if first_token_time is not None:
        ttft_s = first_token_time - start_time
    else:
        ttft_s = total_s

    output_text = "".join(collected_chunks)

    completion_tokens = (
        recorded_usage.completion_tokens
        if recorded_usage and recorded_usage.completion_tokens > 0
        else len(collected_chunks)
    )
    prompt_tokens = (
        recorded_usage.prompt_tokens
        if recorded_usage and recorded_usage.prompt_tokens > 0
        else max(1, len(prompt_text.split()))
    )

    tps = round(completion_tokens / total_s, 2) if total_s > 0 else 0.0

    return BenchmarkRun(
        prompt=prompt_text,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=round(ttft_s, 4),
        total_s=round(total_s, 4),
        tokens_per_second=tps,
        output_text=output_text,
        token_timings=token_timings,
    )


async def run_benchmark_suite(
    client: AsyncLLMClient,
    prompts: list[str],
    settings: list[dict[str, Any]],
    repeats: int = 3,
    delay_between_runs_s: float = 0.0,
) -> list[BenchmarkRun]:
    """Cartesian product of prompts x settings x repeats. Returns all runs."""
    default_model = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b")
    runs: list[BenchmarkRun] = []

    for prompt in prompts:
        for setting in settings:
            model = setting.get("model", default_model)
            temperature = float(setting.get("temperature", 0.7))
            top_p = setting.get("top_p")
            max_tokens = int(setting.get("max_tokens", 256))

            req = LLMRequest(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )

            for _ in range(repeats):
                if delay_between_runs_s > 0 and runs:
                    await asyncio.sleep(delay_between_runs_s)
                run_result = await run_single_benchmark(client, req)
                runs.append(run_result)

    return runs



def summarize(runs: list[BenchmarkRun]) -> dict[str, Any]:
    """Group by (prompt, temperature, top_p) -> avg TTFT, avg tokens/sec,
    and a distinctness measure across repeats (e.g. whether output_text
    was identical across all repeats for that group) to demonstrate
    nondeterminism.
    """
    groups: dict[tuple[str, float, float | None], list[BenchmarkRun]] = defaultdict(list)
    for r in runs:
        groups[(r.prompt, r.temperature, r.top_p)].append(r)

    summary_groups: list[dict[str, Any]] = []
    for (prompt, temp, top_p), group_runs in groups.items():
        n = len(group_runs)
        avg_ttft = sum(r.ttft_s for r in group_runs) / n if n else 0.0
        avg_tps = sum(r.tokens_per_second for r in group_runs) / n if n else 0.0
        avg_prompt_toks = sum(r.prompt_tokens for r in group_runs) / n if n else 0.0
        avg_comp_toks = sum(r.completion_tokens for r in group_runs) / n if n else 0.0

        distinct_texts = {r.output_text.strip() for r in group_runs}
        identical = (len(distinct_texts) <= 1)

        summary_groups.append(
            {
                "prompt": prompt,
                "prompt_char_length": len(prompt),
                "temperature": temp,
                "top_p": top_p,
                "repeats": n,
                "avg_prompt_tokens": round(avg_prompt_toks, 1),
                "avg_completion_tokens": round(avg_comp_toks, 1),
                "avg_ttft_s": round(avg_ttft, 4),
                "avg_ttft_ms": round(avg_ttft * 1000, 1),
                "avg_tokens_per_second": round(avg_tps, 2),
                "distinct_outputs_count": len(distinct_texts),
                "identical_across_repeats": identical,
                "outputs": [r.output_text for r in group_runs],
            }
        )

    return {
        "total_runs": len(runs),
        "groups": summary_groups,
    }
