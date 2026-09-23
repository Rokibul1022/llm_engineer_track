"""Typed input/output contracts for the LLM client.

Keeping these separate from client.py means the schemas can be reused
by the FastAPI layer, the tests, and any future provider adapter
without importing HTTP or retry logic.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling cutoff; defaults to None (provider default). Optional for backwards compatibility.",
    )
    max_tokens: int = Field(default=512, gt=0)
    stream: bool = False



class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    latency_ms: float
    attempts: int = Field(description="How many HTTP attempts it took, including retries.")


class AttemptLog(BaseModel):
    """One entry per HTTP attempt, reported live via the `on_attempt`
    callback. Exists purely for observability/demo purposes — the
    client's return value never depends on this being consumed."""

    attempt: int
    outcome: str  # "success" | "rate_limited" | "provider_error" | "timeout" | "bad_request" | "malformed_response"
    detail: str
    elapsed_ms: float
    will_retry: bool
    delay_before_retry_s: float | None = None
