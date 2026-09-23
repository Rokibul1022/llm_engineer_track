"""Exception hierarchy for the LLM client.

Splitting these out lets callers decide, per exception type, whether a
failure is worth retrying, worth surfacing to a user, or worth paging
someone. It also keeps client.py's retry logic readable: one isinstance
check per branch instead of parsing status codes everywhere.
"""


class LLMClientError(Exception):
    """Base class for every error this client raises."""


class LLMTimeoutError(LLMClientError):
    """The provider did not respond within the configured timeout."""


class LLMRateLimitError(LLMClientError):
    """The provider returned 429. Safe to retry with backoff."""


class LLMProviderError(LLMClientError):
    """The provider returned a 5xx. Safe to retry with backoff."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Provider error {status_code}: {body[:200]}")


class LLMBadRequestError(LLMClientError):
    """The provider returned a 4xx that is NOT 429. Never retry — the
    request itself is invalid and retrying would just waste attempts."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Bad request {status_code}: {body[:200]}")


class LLMMalformedResponseError(LLMClientError):
    """The provider returned 200 but the body didn't match the schema
    we expected. Not retried — a malformed 200 usually repeats."""
