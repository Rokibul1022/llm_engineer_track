# Build Prompt — Week 2 Assignment on top of `llm-client`

> Paste this whole file into your coding agent (Claude Code, Cursor, etc.) with repo
> `Rokibul1022/llm_engineer_track` checked out.
> Work happens on a **new branch, `phase1-week2`**, branched off `phase1-week1`
> — do not commit these changes directly to `phase1-week1`.
> This prompt assumes the agent already has read access to `phase1-week1/llm-client/`
> (the code being extended lives there; only the new branch/commits are "week 2").

---

## Context (what already exists — do not rebuild)

The repo already has a production-grade async LLM client at
`phase1-week1/llm-client/`:

- `llm_client/client.py` — `AsyncLLMClient`: retries (full-jitter backoff), fast-fail
  on 4xx/malformed responses, `complete()` and `stream()` methods, monotonic timing
  via `time.perf_counter()`.
- `llm_client/schemas.py` — Pydantic v2 models: `ChatMessage`, `LLMRequest`
  (`model`, `messages`, `temperature`, `max_tokens`, `stream`), `Usage`
  (`prompt_tokens`, `completion_tokens`, `total_tokens`), `LLMResponse`
  (`content`, `model`, `usage`, `latency_ms`, `attempts`), `AttemptLog`.
- `llm_client/exceptions.py` — typed exception hierarchy
  (`LLMRateLimitError`, `LLMBadRequestError`, `LLMTimeoutError`,
  `LLMProviderError`, `LLMMalformedResponseError`).
- `app.py` — FastAPI gateway exposing `POST /v1/complete`, `POST /v1/stream` (SSE),
  `POST /v1/simulate` (fault injection), `GET /health`, `GET /config`.
  `DEFAULT_MODEL = "openai/gpt-oss-120b"` (from `LLM_DEFAULT_MODEL` env var).
- `streamlit_demo.py` — **the real UI** (`streamlit run streamlit_demo.py`):
  a fault-injection demo page using `SCENARIOS` + `AsyncLLMClient` wired to a
  mock transport, showing retry behavior. This is the file to extend for a
  Week 2 module (see Task 6) — not `static/index.html`.
- `static/index.html` — a secondary browser dashboard that already computes and
  displays **TTFT vs decode-phase latency**, tokens/sec, and cost, client-side
  in JS, driven by SSE event timestamps (see `updatePipelineFirstToken`,
  `completePipelineSuccess`). Reference only — do not build the Week 2 module
  here.
- `tests/test_client.py` — 8 passing tests (respx-mocked, no live network).
- `test_report.json` — existing test-report convention/format to reuse for
  this week's test report.

**Gap for this assignment:** TTFT/decode/tokens-per-second are currently only
computed live in the browser JS for the demo UI. There is no server-side,
reusable, scriptable benchmark that produces a saved report, and there is no
artifact demonstrating *nondeterminism* across `temperature`/`top_p` settings.
There is also no written explanation tying pipeline stages to this codebase.

---

## Goal

Add a **server-independent benchmarking module** and a **pipeline explanation
doc** to `phase1-week1/llm-client/`, reusing `AsyncLLMClient` exactly as-is
(no changes to `client.py` behavior). Deliverable must satisfy: *"Explain the
inference pipeline and benchmark token generation."*

---

## Tasks

### 1. `llm_client/benchmark.py` (new file)

Implement, using **only** `AsyncLLMClient.stream()` (already handles SSE +
retries) — do not reimplement HTTP/retry logic:

```python
from pydantic import BaseModel

class TokenTimingSample(BaseModel):
    token_index: int
    elapsed_since_start_s: float

class BenchmarkRun(BaseModel):
    prompt: str
    model: str
    temperature: float
    top_p: float | None
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    tokens_per_second: float
    output_text: str

async def run_single_benchmark(
    client: "AsyncLLMClient",
    request: "LLMRequest",
) -> BenchmarkRun:
    """Stream one completion, timing first-token latency (TTFT) and
    overall tokens/sec using client.stream(). Reuses LLMRequest/Usage
    schemas already defined in schemas.py — do not redefine them."""
    ...

async def run_benchmark_suite(
    client: "AsyncLLMClient",
    prompts: list[str],
    settings: list[dict],   # e.g. [{"temperature": 0.0}, {"temperature": 0.7}, {"temperature": 1.2}]
    repeats: int = 3,
) -> list[BenchmarkRun]:
    """Cartesian product of prompts x settings x repeats. Returns all runs."""
    ...

def summarize(runs: list[BenchmarkRun]) -> dict:
    """Group by (prompt, temperature, top_p) -> avg TTFT, avg tokens/sec,
    and a distinctness measure across repeats (e.g. whether output_text
    was identical across all repeats for that group) to demonstrate
    nondeterminism."""
    ...
```

Requirements:
- Must use `LLMRequest.temperature` field that already exists in `schemas.py`
  (`Field(default=0.7, ge=0.0, le=2.0)`). If `top_p` isn't already a field on
  `LLMRequest`, add it as an **optional** field (`top_p: float | None = None`)
  with a short docstring note — don't break existing tests, keep it optional
  so existing calls without `top_p` still validate.
- `elapsed_since_start_s` and `ttft_s` must come from `time.perf_counter()`,
  matching the convention already used in `client.py`, not `time.time()`.
- Output must be JSON-serializable (Pydantic `.model_dump()`), written to
  `benchmark_results.json` in the same directory as `test_report.json`
  already sitting in this repo — follow that existing file's naming
  convention.

### 2. `scripts/run_benchmark.py` (new file, or `benchmark_cli.py` at repo root of `llm-client/`)

A small CLI entrypoint:

```bash
python scripts/run_benchmark.py --model openai/gpt-oss-120b --repeats 3
```

- Loads `LLM_API_KEY` / `LLM_BASE_URL` from `.env` (reuse the existing
  `.env.example` pattern already in this repo — do not invent a new env var
  scheme).
- Runs `run_benchmark_suite` against 2–3 prompts of different lengths
  (short/medium/long, to show TTFT scaling with prompt size) and 3 settings:
  `temperature=0.0`, `temperature=0.7`, `temperature=1.2`.
- Prints a summary table to stdout (prompt length, avg TTFT ms, avg
  tokens/sec, "identical across repeats: yes/no").
- Writes full raw results to `benchmark_results.json`.

### 3. `tests/test_benchmark.py` (new file)

Follow the exact style of the existing `tests/test_client.py` (respx-mocked
`httpx.MockTransport`, no live network, async tests). Add at minimum:
- `test_run_single_benchmark_computes_ttft_and_tps` — mock a streamed
  response with known chunk timing, assert TTFT and tokens/sec are
  computed correctly.
- `test_summarize_flags_nondeterminism` — feed `summarize()` a set of runs
  with differing `output_text` at high temperature and identical text at
  `temperature=0.0`, assert the distinctness flag reflects that.

Keep the suite green: run `pytest tests/ -v` and confirm the original 8
tests plus your new ones all pass.

### 4. `PIPELINE.md` (new file, in `phase1-week1/llm-client/`)

A short (not the whole theory — 1 page) explanation document that maps
pipeline stages to **this repo's actual code**, e.g.:

- Tokenization → where `prompt_tokens` in `Usage` comes from (provider-side,
  but note where the client receives/reads it).
- Embeddings/Attention/Logits/Sampling → one paragraph each, generic (this
  repo doesn't implement the model internals — it's a client — say so
  explicitly rather than overclaiming).
- Autoregressive decode loop → tie directly to `client.stream()` in
  `client.py` and the SSE event loop in `app.py`'s `/v1/stream` handler.
- TTFT vs decode-phase latency → tie directly to the existing
  `updatePipelineFirstToken` / `completePipelineSuccess` JS logic in
  `static/index.html`, and to the new server-side `benchmark.py` timings.
- `temperature` / `top_p` and nondeterminism → tie directly to the
  `LLMRequest.temperature` field and the new benchmark suite's
  distinctness-across-repeats result.

Include a short Mermaid diagram (same style as the one already in this
repo's README "System Architecture" ASCII diagram, but as an actual
`mermaid` fenced block) showing: `Prompt → AsyncLLMClient.stream() →
Provider (tokenize→embed→attention→logits→sample→detokenize per token) →
SSE chunks → benchmark.py timing hooks → TTFT/decode/report`.

### 5. Update `README.md` (the one inside `phase1-week1/llm-client/`)

Add one new section, **"Benchmark & Pipeline Explanation"**, right after
the existing "Latency & Cost Decomposition" section, that:
- Links to `PIPELINE.md`.
- Shows the exact `python scripts/run_benchmark.py ...` command.
- Includes a sample of the summary table output (fabricate a plausible
  example row, labeled as an example).

### 6. New Streamlit module: `pages/2_Week2_Benchmark.py`

Convert the UI into Streamlit's native **multipage app** convention (a
`pages/` directory) rather than editing `streamlit_demo.py` in place —
this keeps Week 1's fault-injection demo intact as its own page:

- Move nothing destructive: `streamlit_demo.py` keeps working standalone,
  but also create `pages/1_Week1_Fault_Injection.py` that imports/reuses its
  `SCENARIOS` dict and rendering logic (thin wrapper, no duplicated logic).
- Create `pages/2_Week2_Benchmark.py`:
  - A form: prompt text area, `temperature` slider (0.0–1.5), `top_p` slider,
    "repeats" number input (default 3), model fixed to `openai/gpt-oss-120b`
    (read from `LLM_DEFAULT_MODEL` env, same as `app.py`).
  - A "Run benchmark" button that calls `run_benchmark_suite()` from the new
    `llm_client/benchmark.py` (via `asyncio.run(...)`, matching the async
    pattern already used in `streamlit_demo.py`).
  - Results table: prompt length, TTFT (ms), tokens/sec, "identical across
    repeats: yes/no" — same columns as the CLI summary in Task 2.
  - A small expander showing the raw `output_text` per repeat, so the grader
    can visually see nondeterminism at `temperature=1.2` vs identical output
    at `temperature=0.0`.
  - A short static markdown block at the bottom titled "What this page
    demonstrates" that briefly restates: tokenization → embeddings →
    attention → autoregressive decoding, and temperature/top-p/context-limit
    effects on nondeterminism — 3–4 lines, not a full essay (full explanation
    lives in `PIPELINE.md` / `ANSWERS.md`, link to both).

---

### 7. `LIMITATIONS.md` (new file, in `phase1-week1/llm-client/`)

Short (half a page), honest, specific to this repo — not generic disclaimers:
- **Tokenization/attention/embeddings aren't observable from the client** —
  this repo benchmarks an API boundary (`openai/gpt-oss-120b` via HTTP), not
  the model internals; `prompt_tokens`/`completion_tokens` come from the
  provider's own count, which you don't independently verify.
- **TTFT includes network + provider queueing**, not just model compute —
  say this explicitly, since it means TTFT numbers aren't pure "time to
  produce token 1" in isolation.
- **Nondeterminism check is a sample of 3 repeats** — not statistically
  rigorous; a `temperature=0.0` match across 3 runs doesn't *prove* full
  determinism (providers can still vary due to batching/hardware
  nondeterminism even at temp 0).
- **Streamlit benchmark page runs synchronously in the browser session** —
  note if long benchmark suites (many prompts × many repeats) will block the
  UI thread; call out `asyncio.run()` inside Streamlit as a known constraint,
  not production-grade concurrency.
- One or two more specific to whatever the coding agent actually builds —
  the agent should append real caveats it hits, not just repeat this list.

### 8. `ANSWERS.md` (new file, in `phase1-week1/llm-client/`)

A direct, readable answers doc for the two "must understand this week"
knowledge targets, written so a grader can see understanding at a glance —
not a copy of `PIPELINE.md`, shorter and more direct, in your own words,
referencing this repo's benchmark/results where it strengthens the answer.
See the separate `answers.md` file already provided alongside this prompt
for the expected content/structure — the agent should produce something in
that shape, updated with real numbers once `benchmark_results.json` exists.

---

## Non-goals / constraints

- Do **not** modify `client.py`'s retry/backoff logic — it's already correct
  and tested for Week 1's acceptance matrix.
- Do **not** touch `static/index.html`'s existing dashboard — this is a
  separate, scriptable/reportable path, not a UI change.
- Do **not** add new top-level dependencies beyond what's already in
  `pyproject.toml` / `requirements.txt` unless strictly necessary (e.g. if
  you need `tabulate` for the CLI summary table, prefer plain
  string-formatting instead to avoid a new dependency).
- Keep everything async and typed, matching the existing code style
  (type hints, docstrings in the same tone as `client.py`'s).

## Acceptance criteria (mirror the repo's own test-matrix style)

| # | Check | How to verify |
|---|---|---|
| 1 | `benchmark.py` computes TTFT + tokens/sec from a mocked stream | `pytest tests/test_benchmark.py -v` |
| 2 | CLI script runs end-to-end against `.env` credentials and writes `benchmark_results.json` | manual run |
| 3 | Nondeterminism demonstrated: temp=0.0 repeats identical, temp=1.2 repeats differ | inspect `summarize()` output |
| 4 | `PIPELINE.md` maps every stage to a real file/function in this repo | manual review |
| 5 | All existing 8 tests in `test_client.py` still pass unmodified | `pytest tests/ -v` |
| 6 | README updated with the new section and a working command | manual review |
| 7 | `pages/2_Week2_Benchmark.py` runs the suite live in Streamlit and shows nondeterminism | `streamlit run streamlit_demo.py`, open Week 2 page |
| 8 | `pages/1_Week1_Fault_Injection.py` still reproduces all original scenarios | manual click-through |

---

## Branch, PR & submission checklist

1. Branch off `phase1-week1` → `phase1-week2`. Commit all of the above on it.
2. Open a **pull request** `phase1-week2 → phase1-week1` (or `main`, whichever
   this course track uses as its integration branch) titled something like
   *"Week 2: inference pipeline explanation + token-generation benchmark"*.
   PR description should link `PIPELINE.md`, `ANSWERS.md`, and the new
   Streamlit page, plus paste one sample benchmark table.
3. Submission bundle (per the assignment's own stated requirements) —
   confirm each is present before submitting:
   - [ ] **Pull request** (open, linking the branch diff)
   - [ ] **Test report** — run `pytest tests/ -v`, save output alongside
         `test_report.json` in the same format/location as Week 1's
   - [ ] **Short README** — the new "Benchmark & Pipeline Explanation"
         section added in Task 5 (keep it short, it's a section not a rewrite)
   - [ ] **Architecture diagram** — already produced (`architecture.md`,
         Mermaid diagram) — link it from the README, no need to remake it
   - [ ] **Report explaining tradeoffs and known limitations** — new short
         doc, see Task 7 below
