"""CLI script to run LLM token generation benchmarks across prompt lengths and temperature settings.

Usage:
    python scripts/run_benchmark.py --model openai/gpt-oss-120b --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# Ensure parent directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from llm_client import AsyncLLMClient, run_benchmark_suite, summarize



def load_env_file(env_path: Path) -> None:
    """Simple .env loader without requiring python-dotenv dependency."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = val


PROMPTS = [
    # 1. Short prompt
    "Explain what a token is in 10 words.",
    # 2. Medium prompt
    (
        "Compare self-attention and cross-attention mechanisms in Transformer architectures. "
        "Highlight the primary difference in how keys, queries, and values are sourced."
    ),
    # 3. Long prompt (simulating context processing)
    (
        "Modern autoregressive Transformer inference consists of two distinct computational regimes: "
        "the prefill (prompt processing) phase and the decode (token-by-token generation) phase. "
        "In the prefill phase, all prompt tokens are processed simultaneously in parallel via matrix multiplications, "
        "populating the Key-Value (KV) cache. In the decode phase, each new token requires reading all previous KV states "
        "from High Bandwidth Memory (HBM) to compute attention, rendering single-batch decoding heavily memory-bandwidth bound. "
        "Techniques such as continuous batching, PagedAttention, and Speculative Decoding attempt to maximize compute throughput. "
        "Based on this overview, identify the main bottleneck of the decode phase and how KV caching mitigates recomputation."
    ),
]

SETTINGS = [
    {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
    {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128},
    {"temperature": 1.2, "top_p": 1.0, "max_tokens": 128},
]


def print_ascii_table(summary_groups: list[dict[str, Any]]) -> None:
    headers = [
        "Prompt Size",
        "Temp",
        "Top-P",
        "Runs",
        "Prompt Tok",
        "Comp Tok",
        "TTFT (ms)",
        "Tokens/s",
        "Identical?",
    ]
    col_widths = [14, 6, 7, 6, 12, 10, 11, 10, 12]

    def format_row(cols: list[str]) -> str:
        return " | ".join(val.ljust(w) for val, w in zip(cols, col_widths))

    separator = "-+-".join("-" * w for w in col_widths)

    print("\n" + "=" * len(separator))
    print(" [BENCHMARK] INFERENCE BENCHMARK SUMMARY TABLE")
    print("=" * len(separator))
    print(format_row(headers))
    print(separator)

    for g in summary_groups:
        prompt_len = g["prompt_char_length"]
        if prompt_len < 60:
            size_label = f"Short ({prompt_len}c)"
        elif prompt_len < 200:
            size_label = f"Medium ({prompt_len}c)"
        else:
            size_label = f"Long ({prompt_len}c)"

        top_p_str = str(g["top_p"]) if g["top_p"] is not None else "default"
        identical_str = "YES (det)" if g["identical_across_repeats"] else f"NO ({g['distinct_outputs_count']} var)"

        row = [
            size_label,
            f"{g['temperature']:.1f}",
            top_p_str,
            str(g["repeats"]),
            f"{g['avg_prompt_tokens']:.0f}",
            f"{g['avg_completion_tokens']:.0f}",
            f"{g['avg_ttft_ms']:.1f}",
            f"{g['avg_tokens_per_second']:.1f}",
            identical_str,
        ]
        print(format_row(row))

    print(separator)
    print()


async def main() -> None:
    load_env_file(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Run token generation benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b"),
        help="LLM model identifier",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per prompt/parameter configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON filename relative to project root",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use simulated mock streaming transport instead of live API",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between successive benchmark requests to avoid provider RPM limits",
    )
    args = parser.parse_args()

    api_key = os.getenv("LLM_API_KEY", "demo-key")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    transport = None
    if args.mock:
        import httpx

        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            prompt_content = body["messages"][-1]["content"]
            temp = body.get("temperature", 0.7)

            if temp == 0.0:
                answer = f"Deterministic response to '{prompt_content[:20]}...'"
            else:
                import uuid
                answer = f"Varied response [{uuid.uuid4().hex[:6]}] to '{prompt_content[:20]}...'"

            words = answer.split()
            lines = []
            for w in words:
                lines.append(f'data: {{"choices": [{{"delta": {{"content": " {w}"}}}}]}}\n\n')
            lines.append(f'data: {{"usage": {{"prompt_tokens": {len(prompt_content.split()) + 4}, "completion_tokens": {len(words)}, "total_tokens": {len(prompt_content.split()) + 4 + len(words)}}}}}\n\n')
            lines.append("data: [DONE]\n\n")

            return httpx.Response(
                200,
                text="".join(lines),
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(mock_handler)
        print("[MOCK] Running in simulated mock transport mode.")

    client = AsyncLLMClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,
        base_delay_s=0.5,
        max_delay_s=3.0,
        transport=transport,
    )

    settings = [dict(s, model=args.model) for s in SETTINGS]
    pace_delay = 0.0 if args.mock else args.delay

    print(f"[START] Starting benchmark suite: {len(PROMPTS)} prompts x {len(settings)} settings x {args.repeats} repeats")
    print(f"        Model: {args.model}")
    print(f"        Base URL: {base_url}")
    print(f"        Pacing delay: {pace_delay:.1f}s between requests")
    print("        Benchmarking TTFT, Tokens/sec, and Temperature-based nondeterminism...\n")

    try:
        runs = await run_benchmark_suite(
            client=client,
            prompts=PROMPTS,
            settings=settings,
            repeats=args.repeats,
            delay_between_runs_s=pace_delay,
        )
    finally:
        await client.aclose()


    summary = summarize(runs)
    print_ascii_table(summary["groups"])

    output_path = PROJECT_DIR / args.output
    payload = {
        "metadata": {
            "model": args.model,
            "repeats": args.repeats,
            "total_runs": len(runs),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "summary": summary,
        "runs": [r.model_dump() for r in runs],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SAVED] Raw benchmark results saved to {output_path.resolve()}")



if __name__ == "__main__":
    asyncio.run(main())
