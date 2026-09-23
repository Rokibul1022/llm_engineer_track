"""Groq API client wrapper supporting tool calling and streaming."""
import json
import os
from typing import Any, Generator
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Load credentials from active directory, phase1-week3, or phase1-week1/llm-client
for env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / "phase1-week1" / "llm-client" / ".env",
    Path(__file__).parent.parent / "phase1-week1" / ".env",
    Path.cwd() / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path, override=False)

DEFAULT_BASE_URL = os.environ.get("GROQ_BASE_URL", os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", os.environ.get("LLM_DEFAULT_MODEL", "openai/gpt-oss-120b"))


class FunctionCall:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


class ToolCall:
    def __init__(self, id: str = "", type: str = "function", function: FunctionCall | None = None):
        self.id = id
        self.type = type
        self.function = function or FunctionCall()

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "function": self.function.to_dict()}


class Message:
    def __init__(
        self,
        role: str = "assistant",
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class Choice:
    def __init__(
        self,
        index: int = 0,
        message: Message | None = None,
        delta: Message | None = None,
        finish_reason: str | None = None,
    ):
        self.index = index
        self.message = message or Message()
        self.delta = delta or Message()
        self.finish_reason = finish_reason


class GroqResponse:
    def __init__(self, choices: list[Choice]):
        self.choices = choices


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY or LLM_API_KEY environment variable is required.")
    return key


def call_groq(
    messages: list[dict],
    tool_def: dict,
    tool_name: str,
    stream: bool = False,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> GroqResponse | Generator[GroqResponse, None, None]:
    """Call Groq chat completion with tool calling, returning GroqResponse or streaming chunks."""
    api_key = get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": [tool_def],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
        "temperature": temperature,
        "stream": stream,
    }

    if not stream:
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            err_data = {}
            try:
                err_data = resp.json()
            except Exception:
                pass
            err_obj = err_data.get("error", {}) if isinstance(err_data, dict) else {}
            # Handle Groq's tool_use_failed by feeding failed generation into the validator/retry loop
            if resp.status_code == 400 and err_obj.get("code") == "tool_use_failed":
                failed_gen = err_obj.get("failed_generation", "")
                if failed_gen:
                    try:
                        fg_obj = json.loads(failed_gen)
                        args = fg_obj.get("arguments", {})
                        arg_str = json.dumps(args) if not isinstance(args, str) else args
                        tc = ToolCall(
                            id="call_failed_gen",
                            type="function",
                            function=FunctionCall(name=fg_obj.get("name", tool_name), arguments=arg_str),
                        )
                        msg = Message(role="assistant", tool_calls=[tc])
                        return GroqResponse(choices=[Choice(index=0, message=msg)])
                    except Exception:
                        pass
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

        data = resp.json()
        choices: list[Choice] = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            raw_tools = msg_data.get("tool_calls") or []
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    ),
                )
                for tc in raw_tools
            ]
            msg = Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
                tool_calls=tool_calls,
            )
            choices.append(Choice(index=c.get("index", 0), message=msg, finish_reason=c.get("finish_reason")))

        return GroqResponse(choices=choices)

    else:
        def stream_generator() -> Generator[GroqResponse, None, None]:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        err_body = response.read().decode(errors="replace")
                        raise RuntimeError(f"Groq API streaming error {response.status_code}: {err_body}")

                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk_str = line[6:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            continue

                        choices: list[Choice] = []
                        for c in chunk_data.get("choices", []):
                            delta_data = c.get("delta", {})
                            raw_tools = delta_data.get("tool_calls") or []
                            tool_calls = [
                                ToolCall(
                                    id=tc.get("id", ""),
                                    type=tc.get("type", "function"),
                                    function=FunctionCall(
                                        name=tc.get("function", {}).get("name", ""),
                                        arguments=tc.get("function", {}).get("arguments", ""),
                                    ),
                                )
                                for tc in raw_tools
                            ]
                            delta_msg = Message(
                                role=delta_data.get("role", "assistant"),
                                content=delta_data.get("content"),
                                tool_calls=tool_calls,
                            )
                            choices.append(Choice(index=c.get("index", 0), delta=delta_msg, finish_reason=c.get("finish_reason")))

                        yield GroqResponse(choices=choices)

        return stream_generator()
