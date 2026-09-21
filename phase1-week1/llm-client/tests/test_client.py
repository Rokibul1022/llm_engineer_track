"""Tests use respx to mock the HTTP layer — no real API key or network
call needed. Each test targets one failure mode named in the
assignment's acceptance test: rate limits, malformed outputs,
timeouts, and provider failures, plus the "no uncontrolled retries"
requirement (a 4xx must fail on the first attempt).
"""
import httpx
import pytest
import respx

from llm_client import (
    AsyncLLMClient,
    ChatMessage,
    LLMBadRequestError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMRequest,
)

URL = "https://api.example.com/v1/chat/completions"


def make_request() -> LLMRequest:
    return LLMRequest(model="gpt-test", messages=[ChatMessage(role="user", content="hi")])


def ok_body(content: str = "hello back") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


@pytest.fixture
def client():
    c = AsyncLLMClient(api_key="test-key", base_url="https://api.example.com/v1", max_retries=3, base_delay_s=0.01, max_delay_s=0.02)
    yield c


@pytest.mark.asyncio
@respx.mock
async def test_success_on_first_try(client):
    respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    result = await client.complete(make_request())
    assert result.content == "hello back"
    assert result.attempts == 1
    assert result.usage.total_tokens == 8


@pytest.mark.asyncio
@respx.mock
async def test_retries_then_succeeds_after_500(client):
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(500, text="server error"),
        httpx.Response(500, text="server error"),
        httpx.Response(200, json=ok_body()),
    ]
    result = await client.complete(make_request())
    assert result.attempts == 3
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_exhausts_retries_and_raises(client):
    respx.post(URL).mock(return_value=httpx.Response(429, text="rate limited"))
    with pytest.raises(LLMRateLimitError):
        await client.complete(make_request())


@pytest.mark.asyncio
@respx.mock
async def test_bad_request_fails_immediately_no_retry(client):
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad request"))
    with pytest.raises(LLMBadRequestError):
        await client.complete(make_request())
    # This is the "no uncontrolled retries" acceptance criterion:
    # a 4xx must not be retried.
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_timeout_is_retried_then_raises(client):
    respx.post(URL).mock(side_effect=httpx.TimeoutException("timed out"))
    with pytest.raises(Exception):
        await client.complete(make_request())
    assert respx.calls.call_count == client._max_retries


@pytest.mark.asyncio
@respx.mock
async def test_malformed_200_raises_without_retry(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))
    with pytest.raises(LLMMalformedResponseError):
        await client.complete(make_request())
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_stream_success(client):
    sse_data = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " world!"}}]}\n\n'
        'data: {"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}\n\n'
        'data: [DONE]\n\n'
    )
    respx.post(URL).mock(
        return_value=httpx.Response(200, text=sse_data, headers={"content-type": "text/event-stream"})
    )

    tokens = []
    usages = []
    async for token in client.stream(make_request(), on_usage=usages.append):
        tokens.append(token)

    assert "".join(tokens) == "Hello world!"
    assert len(usages) == 1
    assert usages[0].total_tokens == 6


@pytest.mark.asyncio
@respx.mock
async def test_stream_rate_limit_retries_and_raises(client):
    respx.post(URL).mock(return_value=httpx.Response(429, text="rate limited"))
    with pytest.raises(LLMRateLimitError):
        async for _ in client.stream(make_request()):
            pass

