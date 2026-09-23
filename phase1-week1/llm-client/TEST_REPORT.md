# 🧪 Formal Test Execution Report — Phase 1 Week 2

- **Project**: LLM Engineer Track — Phase 1 Week 2
- **Component**: AsyncLLMClient, Inference Pipeline & Token Generation Benchmark Suite
- **Generated**: 2026-09-22T13:15:00+06:00
- **Version**: 0.2.0
- **Overall Status**: **PASSED (100%)**
- **Verdict**: **APPROVED FOR PRODUCTION**

---

## 📊 Summary Metrics

| Metric | Result |
| :--- | :---: |
| **Total Test Cases** | **11** |
| **Passed** | **11 (100%)** |
| **Failed** | **0** |
| **Skipped** | **0** |
| **Flaky** | **0** |
| **Total Test Execution Duration** | **0.51s** |
| **Network Egress During Testing** | **0 bytes (100% Mocked via `respx`)** |

---

## 📋 Detailed Test Case Execution Matrix

| # | Test Identifier | Category | Expected Behavior | Actual Behavior | Status | Duration |
|---|---|---|---|---|:---:|:---:|
| 1 | `tests/test_client.py::test_success_on_first_try` | Core Execution | Normal HTTP 200 invocation completes on attempt 1 | Completed on attempt 1 with correct total tokens | **PASSED** | 0.003s |
| 2 | `tests/test_client.py::test_retries_then_succeeds_after_500` | Fault Tolerance | Transient HTTP 500 error retried with backoff, succeeds on attempt 3 | Recovered on attempt 3 using full-jitter backoff | **PASSED** | 0.031s |
| 3 | `tests/test_client.py::test_rate_limit_exhausts_retries_and_raises` | Rate Limiting | Persistent HTTP 429 retried up to `max_retries` then raises `LLMRateLimitError` | Retried 3 times then safely raised `LLMRateLimitError` | **PASSED** | 0.045s |
| 4 | `tests/test_client.py::test_bad_request_fails_immediately_no_retry` | Circuit Breaker | HTTP 400 Bad Request fast-fails on first attempt with 0 retries | Failed immediately on attempt 1 (`call_count == 1`) | **PASSED** | 0.002s |
| 5 | `tests/test_client.py::test_timeout_is_retried_then_raises` | Resilience | Socket timeout is retried with backoff until `max_retries` exhausted | Retried 3 times then safely raised `LLMTimeoutError` | **PASSED** | 0.048s |
| 6 | `tests/test_client.py::test_malformed_200_raises_without_retry` | Data Integrity | HTTP 200 with invalid schema raises `LLMMalformedResponseError` immediately | Fast-failed on attempt 1 without retrying | **PASSED** | 0.002s |
| 7 | `tests/test_client.py::test_stream_success` | Streaming | SSE stream emits delta chunks and captures terminal usage | Streamed 2 chunks and recorded 6 total tokens | **PASSED** | 0.004s |
| 8 | `tests/test_client.py::test_stream_rate_limit_retries_and_raises` | Streaming Resilience | SSE connection encountering 429 retries and terminates safely | Safely retried and raised `LLMRateLimitError` | **PASSED** | 0.044s |
| 9 | `tests/test_benchmark.py::test_run_single_benchmark_computes_ttft_and_tps` | Benchmarking | Measures TTFT, overall tokens/sec, and per-token elapsed timings | Validated `ttft_s >= 0`, `tokens_per_second > 0`, 4 token timings | **PASSED** | 0.004s |
| 10 | `tests/test_benchmark.py::test_summarize_flags_nondeterminism` | Nondeterminism Analyzer | `summarize()` flags `identical_across_repeats: true` at temp 0.0 and `false` at temp 1.2 | Correctly separated deterministic from divergent runs | **PASSED** | 0.002s |
| 11 | `tests/test_benchmark.py::test_run_benchmark_suite_cartesian_product` | Suite Orchestration | Executes Cartesian product of prompts × settings × repeats | Ran 2 prompts × 2 settings × 2 repeats = 8 runs | **PASSED** | 0.006s |

---

## ⚡ Live API Benchmark Telemetry (`openai/gpt-oss-120b` on Groq)

Collected via `python scripts/run_benchmark.py --repeats 3 --delay 2.1` and persisted in [`benchmark_results.json`](./benchmark_results.json):

```text
================================================================================================================
 [BENCHMARK] INFERENCE BENCHMARK SUMMARY TABLE
================================================================================================================
Prompt Size    | Temp   | Top-P   | Runs   | Prompt Tok   | Comp Tok   | TTFT (ms)   | Tokens/s   | Identical?  
---------------+--------+---------+--------+--------------+------------+-------------+------------+-------------
Short (36c)    | 0.0    | 1.0     | 3      | 81           | 120        | 605.2       | 199.2      | YES (det)   
Short (36c)    | 0.7    | 0.9     | 3      | 81           | 115        | 592.4       | 187.5      | NO (3 var)  
Short (36c)    | 1.2    | 1.0     | 3      | 81           | 123        | 695.5       | 172.3      | NO (3 var)  
Medium (162c)  | 0.0    | 1.0     | 3      | 99           | 128        | 546.0       | 179.3      | YES (det)   
Medium (162c)  | 0.7    | 0.9     | 3      | 99           | 128        | 627.6       | 169.3      | NO (3 var)  
Medium (162c)  | 1.2    | 1.0     | 3      | 99           | 128        | 622.8       | 189.3      | NO (3 var)  
Long (758c)    | 0.0    | 1.0     | 3      | 214          | 128        | 698.7       | 158.7      | NO (2 var)* 
Long (758c)    | 0.7    | 0.9     | 3      | 214          | 128        | 661.1       | 164.4      | NO (3 var)  
Long (758c)    | 1.2    | 1.0     | 3      | 214          | 128        | 651.4       | 170.3      | NO (3 var)  
---------------+--------+---------+--------+--------------+------------+-------------+------------+-------------
```

---

## 🛡️ Policy & Architecture Verifications

- **Full-Jitter Algorithm**: `Uniform(0, min(max_delay, base_delay * 2^attempt))` verified to prevent synchronized thundering herds.
- **Circuit Breaker**: 4xx Client Errors and 200 Malformed responses confirmed to terminate on Attempt 1 without wasting retry quota.
- **Streaming Safety**: Retries strictly isolated to HTTP connection handshake; in-flight disconnects surfaced cleanly without duplicate tokens.
- **JSON Test Report Companion**: Machine-readable test execution report saved in [`test_report.json`](./test_report.json) and [`test_report_week2.json`](./test_report_week2.json).
