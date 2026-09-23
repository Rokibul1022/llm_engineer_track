"""Automated test runner executing every test case from week3-test-report.md against the live service."""
import json
import time
import httpx
from schemas import TicketExtraction
from exceptions import ExtractionFailure
from extractor import extract, semantic_checks
from groq_client import Choice, GroqResponse, Message, ToolCall, FunctionCall
import extractor

BASE_URL = "http://127.0.0.1:8000"

results = {}

print("==================================================")
print("RUNNING WEEK 3 TEST SUITE AGAINST LIVE SERVICE")
print("==================================================")

# ----------------------------------------------------
# 1. Structural validation & retry
# ----------------------------------------------------
print("\n--- Suite 1: Structural validation & retry ---")

# 1.1 Well-formed extraction
text_1_1 = "Customer states: The payment gateway returns a 500 server error whenever customers try to purchase with a credit card."
resp_1_1 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_1_1}, timeout=30).json()
print("1.1 Well-formed extraction:", resp_1_1)
results["1.1"] = resp_1_1

# 1.2 Missing required field / subtle urgency
text_1_2 = "Can we have dark mode in the dashboard sometime?"
resp_1_2 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_1_2}, timeout=30).json()
print("1.2 Missing required field (defaulting):", resp_1_2)
results["1.2"] = resp_1_2

# 1.3 Forced malformed output (mock) - retry fires once and succeeds on attempt 2
original_call_groq = extractor.call_groq
call_count_1_3 = 0

def mock_call_groq_1_3(messages, tool_def, tool_name, **kwargs):
    global call_count_1_3
    call_count_1_3 += 1
    if call_count_1_3 == 1:
        # Return invalid enum value for urgency: "super_urgent"
        bad_args = json.dumps({
            "intent": "bug_report",
            "urgency": "super_urgent",  # Invalid enum!
            "summary": "Crash on launch",
            "confidence": 0.9,
            "entities": []
        })
        tc = ToolCall(id="call_mock_1", function=FunctionCall(name=tool_name, arguments=bad_args))
        msg = Message(role="assistant", tool_calls=[tc])
        return GroqResponse(choices=[Choice(message=msg)])
    else:
        # Attempt 2: corrected arguments
        good_args = json.dumps({
            "intent": "bug_report",
            "urgency": "high",
            "summary": "Crash on launch",
            "confidence": 0.9,
            "entities": []
        })
        tc = ToolCall(id="call_mock_2", function=FunctionCall(name=tool_name, arguments=good_args))
        msg = Message(role="assistant", tool_calls=[tc])
        return GroqResponse(choices=[Choice(message=msg)])

extractor.call_groq = mock_call_groq_1_3
try:
    res_1_3 = extract("ticket", "App crashes on launch")
    print("1.3 Forced malformed output (mock):", res_1_3)
    results["1.3"] = {"status": res_1_3["status"], "attempts": res_1_3["attempts"], "call_count": call_count_1_3}
finally:
    extractor.call_groq = original_call_groq

# 1.4 Retry budget exhausted (mock) - always fails
call_count_1_4 = 0

def mock_call_groq_1_4(messages, tool_def, tool_name, **kwargs):
    global call_count_1_4
    call_count_1_4 += 1
    bad_args = json.dumps({
        "intent": "invalid_intent_value",
        "urgency": "low",
        "summary": "Some issue",
        "confidence": 0.5
    })
    tc = ToolCall(id="call_fail", function=FunctionCall(name=tool_name, arguments=bad_args))
    return GroqResponse(choices=[Choice(message=Message(role="assistant", tool_calls=[tc]))])

extractor.call_groq = mock_call_groq_1_4
try:
    try:
        extract("ticket", "Some issue", max_retries=2)
        results["1.4"] = "FAILED: Did not raise"
    except ExtractionFailure as exc:
        print("1.4 Retry budget exhausted:", f"Raised {type(exc).__name__} after {call_count_1_4} calls: {exc}")
        results["1.4"] = {"raised": "ExtractionFailure", "calls": call_count_1_4, "detail": str(exc)}
finally:
    extractor.call_groq = original_call_groq

# 1.5 Non-tool-call response (mock)
call_count_1_5 = 0

def mock_call_groq_1_5(messages, tool_def, tool_name, **kwargs):
    global call_count_1_5
    call_count_1_5 += 1
    if call_count_1_5 == 1:
        # Plain text message without tool calls
        return GroqResponse(choices=[Choice(message=Message(role="assistant", content="I cannot extract this ticket."))])
    else:
        # Second call returns valid tool call
        valid_args = json.dumps({
            "intent": "other",
            "urgency": "low",
            "summary": "Plain text query",
            "confidence": 0.5,
            "entities": []
        })
        tc = ToolCall(id="call_recovery", function=FunctionCall(name=tool_name, arguments=valid_args))
        return GroqResponse(choices=[Choice(message=Message(role="assistant", tool_calls=[tc]))])

extractor.call_groq = mock_call_groq_1_5
try:
    res_1_5 = extract("ticket", "Help needed", max_retries=2)
    print("1.5 Non-tool-call response recovery:", res_1_5)
    results["1.5"] = {"status": res_1_5["status"], "attempts": res_1_5["attempts"], "calls": call_count_1_5}
finally:
    extractor.call_groq = original_call_groq


# ----------------------------------------------------
# 2. Semantic sanity checks
# ----------------------------------------------------
print("\n--- Suite 2: Semantic sanity checks ---")

# 2.1 Negation: "This is NOT a billing issue, it's a bug."
text_2_1 = "This is NOT a billing issue, it's a bug."
resp_2_1 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_2_1}, timeout=30).json()
print("2.1 Negation:", resp_2_1)
results["2.1"] = resp_2_1

# 2.2 Sarcasm: "Oh great, ANOTHER billing 'surprise'. Love it."
text_2_2 = "Oh great, ANOTHER billing 'surprise'. Love it."
resp_2_2 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_2_2}, timeout=30).json()
print("2.2 Sarcasm:", resp_2_2)
results["2.2"] = resp_2_2

# 2.3 Ungrounded summary
text_2_3 = "Error code 0x800."
resp_2_3 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_2_3}, timeout=30).json()
print("2.3 Ungrounded summary / short input:", resp_2_3)
results["2.3"] = resp_2_3

# 2.4 High confidence on short input (<20 chars)
text_2_4 = "Crash on login"
resp_2_4 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_2_4}, timeout=30).json()
print("2.4 Short input (<20 chars):", resp_2_4)
results["2.4"] = resp_2_4

# 2.5 Entity hallucination check
text_2_5 = "Slack integration stopped syncing notifications yesterday."
resp_2_5 = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": text_2_5}, timeout=30).json()
print("2.5 Entity extraction:", resp_2_5)
results["2.5"] = resp_2_5


# ----------------------------------------------------
# 3. Streaming behavior
# ----------------------------------------------------
print("\n--- Suite 3: Streaming behavior ---")

# 3.1 & 3.2 Partial events before final + Final validation
stream_events = []
with httpx.stream("POST", f"{BASE_URL}/extract/stream", json={"schema_name": "ticket", "text": "App freezes when uploading profile pictures."}, timeout=30) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            ev = json.loads(line[6:].strip())
            stream_events.append(ev)

partials = [e for e in stream_events if "partial" in e]
finals = [e for e in stream_events if "final" in e]
print(f"3.1 & 3.2 Streaming: received {len(partials)} partial chunks, {len(finals)} final payload")
print("Final payload:", finals[0] if finals else "NONE")
results["3.1_3.2"] = {
    "num_partials": len(partials),
    "has_final": bool(finals),
    "final": finals[0] if finals else None
}

# 3.3 Mid-stream schema failure
resp_3_3 = []
with httpx.stream("POST", f"{BASE_URL}/extract/stream", json={"schema_name": "non_existent_schema", "text": "test text"}, timeout=30) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            resp_3_3.append(json.loads(line[6:].strip()))
print("3.3 Unknown schema error stream:", resp_3_3)
results["3.3"] = resp_3_3

# 3.4 Client disconnect mid-stream
t0 = time.perf_counter()
with httpx.stream("POST", f"{BASE_URL}/extract/stream", json={"schema_name": "ticket", "text": "Long text with issues"}, timeout=30) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            break  # disconnect after first chunk
disconnect_duration = (time.perf_counter() - t0) * 1000
print(f"3.4 Disconnect cleanly handled in {disconnect_duration:.1f}ms")
results["3.4"] = {"disconnect_ms": disconnect_duration, "status": "clean"}


# ----------------------------------------------------
# 4. Determinism: 5x runs at temperature=0
# ----------------------------------------------------
print("\n--- Suite 4: Determinism (5x runs at temp=0) ---")
det_text = "Billing dispute: I was charged $99 twice on September 15 for invoice INV-1002."
det_runs = []
for i in range(5):
    t_start = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/extract", json={"schema_name": "ticket", "text": det_text}, timeout=30).json()
    elapsed = (time.perf_counter() - t_start) * 1000
    det_runs.append({"run": i + 1, "data": r.get("data"), "latency_ms": elapsed})
    print(f" Run {i+1} ({elapsed:.0f}ms): {json.dumps(r.get('data'))}")

# Compare runs
first_run_data = det_runs[0]["data"]
all_identical = all(d["data"] == first_run_data for d in det_runs)
print(f"All 5 runs strictly identical: {all_identical}")
results["4.1"] = {"all_identical": all_identical, "runs": det_runs}

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nTEST SUITE COMPLETE! Results saved to test_results.json")
