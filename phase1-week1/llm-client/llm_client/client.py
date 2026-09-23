"""A small, typed, async LLM client.

Design decisions (also explained in the README):
  - Only 429 and 5xx are retried. 4xx (bad request, auth, etc.) fails
    fast, because retrying a malformed request just burns your rate
    limit budget on a request that will never succeed.
  - Backoff is exponential with full jitter (not fixed-interval), so
    that many concurrent callers hitting a rate limit at once don't
    all retry in lockstep and re-trigger the limit together.
  - Every call reports how many attempts it took and how long it took,
    so callers/observability can tell "slow but healthy" apart from
    "silently retried 4 times."
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable

import httpx

from .exceptions import (
    LLMBadRequestError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .schemas import AttemptLog, LLMRequest, LLMResponse, Usage

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
OnAttempt = Callable[[AttemptLog], None]


class AsyncLLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`transport` is normally left as None (real network). Passing
        an `httpx.MockTransport` here lets tests and the fault-injection
        demo simulate rate limits, timeouts, and malformed responses
        without any real provider — see streamlit_demo.py."""
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncLLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _backoff_delay(self, attempt: int) -> float:
        """Full-jitter exponential backoff: random value between 0 and
        the exponential cap, not a fixed fraction of it. This spreads
        retries out instead of having every client wake up at once."""
        cap = min(self._max_delay_s, self._base_delay_s * (2 ** attempt))
        return random.uniform(0, cap)

    async def complete(self, request: LLMRequest, on_attempt: OnAttempt | None = None) -> LLMResponse:
        """Non-streaming completion with retries and timing.

        `on_attempt`, if given, is called once per HTTP attempt with an
        AttemptLog — before the backoff sleep, so a UI can render each
        attempt live and then watch the actual delay happen."""
        start = time.perf_counter()
        payload = request.model_dump(exclude={"stream"})
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            elapsed_ms = (time.perf_counter() - start) * 1000
            outcome: str
            detail: str
            terminal_error: Exception | None = None

            try:
                resp = await self._client.post(f"{self._base_url}/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(str(exc))
                outcome, detail = "timeout", str(exc)
            except httpx.HTTPError as exc:
                last_error = LLMProviderError(status_code=0, body=str(exc))
                outcome, detail = "provider_error", str(exc)
            else:
                if resp.status_code == 429:
                    last_error = LLMRateLimitError(resp.text)
                    outcome, detail = "rate_limited", resp.text
                elif resp.status_code in RETRYABLE_STATUS:
                    last_error = LLMProviderError(resp.status_code, resp.text)
                    outcome, detail = "provider_error", f"HTTP {resp.status_code}: {resp.text}"
                elif 400 <= resp.status_code < 500:
                    # Not retryable — fail immediately, don't waste attempts.
                    terminal_error = LLMBadRequestError(resp.status_code, resp.text)
                    outcome, detail = "bad_request", f"HTTP {resp.status_code}: {resp.text}"
                else:
                    try:
                        result = self._parse_response(resp, request.model, start, attempt)
                    except LLMMalformedResponseError as exc:
                        terminal_error = exc
                        outcome, detail = "malformed_response", str(exc)
                    else:
                        if on_attempt:
                            on_attempt(AttemptLog(
                                attempt=attempt, outcome="success", detail="OK",
                                elapsed_ms=elapsed_ms, will_retry=False,
                            ))
                        return result

            if terminal_error is not None:
                if on_attempt:
                    on_attempt(AttemptLog(
                        attempt=attempt, outcome=outcome, detail=detail,
                        elapsed_ms=elapsed_ms, will_retry=False,
                    ))
                raise terminal_error

            will_retry = attempt < self._max_retries
            delay = self._backoff_delay(attempt) if will_retry else None
            if on_attempt:
                on_attempt(AttemptLog(
                    attempt=attempt, outcome=outcome, detail=detail,
                    elapsed_ms=elapsed_ms, will_retry=will_retry, delay_before_retry_s=delay,
                ))
            if will_retry:
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    async def stream(
        self, request: LLMRequest, on_usage: Callable[[Usage], None] | None = None
    ) -> AsyncIterator[str]:
        """Streaming completion. Retries only apply to establishing the
        connection — once tokens start flowing, a mid-stream drop is
        surfaced to the caller rather than silently restarted, since a
        partial answer already reached the user.

        `on_usage`, if given, is called once with the final token usage
        — OpenAI-compatible providers (including Groq) can attach usage
        to the terminal chunk when `stream_options.include_usage` is set,
        which is what lets a UI show real input/output token counts for
        a streamed response instead of only the token text."""
        payload = request.model_dump()
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._client.stream(
                    "POST", f"{self._base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status_code == 429:
                        last_error = LLMRateLimitError(await resp.aread())
                    elif resp.status_code in RETRYABLE_STATUS:
                        last_error = LLMProviderError(resp.status_code, (await resp.aread()).decode())
                    elif 400 <= resp.status_code < 500:
                        raise LLMBadRequestError(resp.status_code, (await resp.aread()).decode())
                    else:
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line.removeprefix("data: ")
                            if data.strip() == "[DONE]":
                                return
                            chunk = json.loads(data)
                            if chunk.get("usage") and on_usage:
                                on_usage(Usage(**chunk["usage"]))
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {}).get("content")
                            if delta:
                                yield delta
                        return
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(str(exc))

            if attempt < self._max_retries:
                await asyncio.sleep(self._backoff_delay(attempt))

        assert last_error is not None
        raise last_error

    def _parse_response(
        self, resp: httpx.Response, model: str, start: float, attempts: int
    ) -> LLMResponse:
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            usage = Usage(**body["usage"])
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise LLMMalformedResponseError(f"Unexpected response shape: {exc}") from exc

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=(time.perf_counter() - start) * 1000,
            attempts=attempts,
        )
