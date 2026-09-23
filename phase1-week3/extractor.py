"""Extraction service orchestrating prompt construction, LLM call, validation, and self-correction retry."""
import json
import re
from pydantic import ValidationError
from schemas import SCHEMA_REGISTRY
from prompts import PROMPTS, tool_def_for
from groq_client import call_groq
from exceptions import ExtractionFailure


def extract(schema_name: str, raw_text: str, max_retries: int = 2, force_initial_fault: bool = False, **kwargs) -> dict:
    """Extract structured data matching schema_name with retry on validation failures."""
    if schema_name not in SCHEMA_REGISTRY:
        raise ExtractionFailure(f"Unknown schema '{schema_name}'. Valid options: {list(SCHEMA_REGISTRY)}")

    model_cls = SCHEMA_REGISTRY[schema_name]
    tool_def = tool_def_for(schema_name, model_cls)
    tool_name = f"extract_{schema_name}_data"
    messages: list[dict] = [
        {"role": "system", "content": PROMPTS[schema_name]},
        {"role": "user", "content": raw_text},
    ]

    attempts_used = 0
    attempt_errors: list[str] = []
    attempt_logs: list[dict] = []

    for attempt in range(max_retries + 1):
        attempts_used += 1

        if force_initial_fault and attempt == 0:
            # Simulate initial tool-call with schema violation to demonstrate self-correction recovery
            if schema_name == "ticket":
                simulated_args = json.dumps({
                    "intent": "bug_report",
                    "urgency": "SUPER_CRITICAL_INVALID",
                    "summary": raw_text[:80],
                    "confidence": 0.95
                })
            else:
                simulated_args = json.dumps({
                    "vendor": "Acme Corp",
                    "invoice_number": "INV-001",
                    "amount": "NOT_A_FLOAT",
                    "currency": "BITCOIN",
                    "due_date": "not specified",
                    "confidence": 0.95
                })
            tool_call_id = "call_simulated_fault"
            try:
                args = json.loads(simulated_args)
                model_cls.model_validate(args)
            except ValidationError as sim_err:
                err_str = str(sim_err)
                attempt_errors.append(f"Attempt 1 ValidationError: {err_str}")
                attempt_logs.append({
                    "attempt": 1,
                    "status": "failed",
                    "error": err_str,
                    "arguments": simulated_args
                })
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": simulated_args},
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Validation error: {err_str}. Correct the arguments and call the tool again.",
                })
                continue

        resp = call_groq(messages, tool_def, tool_name=tool_name)
        tool_calls = getattr(resp.choices[0].message, "tool_calls", None) or []

        if not tool_calls:
            err_msg = "Model responded without calling the required extraction tool."
            attempt_errors.append(f"Attempt {attempts_used}: {err_msg}")
            attempt_logs.append({"attempt": attempts_used, "status": "failed", "error": err_msg})
            if attempt == max_retries:
                raise ExtractionFailure(f"Failed after {attempts_used} attempts: {err_msg}")
            messages.append({"role": "assistant", "content": resp.choices[0].message.content or "No tool call"})
            messages.append({
                "role": "user",
                "content": f"Error: You must call the {tool_name} tool with the extracted data. Do not reply with plain text."
            })
            continue

        tool_call = tool_calls[0]
        try:
            raw_args = tool_call.function.arguments
            args = json.loads(raw_args)
            result = model_cls.model_validate(args)
            status = "success" if attempts_used == 1 else "success_after_retry"
            attempt_logs.append({"attempt": attempts_used, "status": "passed", "arguments": raw_args})
            return {
                "status": status,
                "attempts": attempts_used,
                "data": result.model_dump(),
                "attempt_errors": attempt_errors,
                "attempt_logs": attempt_logs,
            }
        except (json.JSONDecodeError, ValidationError) as e:
            err_str = str(e)
            attempt_errors.append(f"Attempt {attempts_used} ValidationError: {err_str}")
            attempt_logs.append({"attempt": attempts_used, "status": "failed", "error": err_str, "arguments": tool_call.function.arguments})
            if attempt == max_retries:
                raise ExtractionFailure(f"Failed after {attempts_used} attempts: {e}")
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Validation error: {e}. Correct the arguments and call the tool again.",
            })


def semantic_checks(result: dict, raw_text: str) -> list[str]:
    """Run non-LLM heuristic checks to flag schema-valid but potentially erroneous extractions."""
    warnings: list[str] = []
    text_lower = raw_text.lower()

    # Generic check: high confidence on very short input
    conf = result.get("confidence", 0.0)
    if conf > 0.9 and len(raw_text.strip()) < 20:
        warnings.append("High confidence on very short input — suspicious.")

    # Ticket-specific checks
    if "intent" in result:
        # Negation check: e.g. "not a billing issue"
        has_negation = bool(re.search(r"\b(not\s+(a\s+)?billing|isn'?t\s+billing)\b", text_lower))
        if has_negation and result.get("intent") == "billing":
            warnings.append("Intent extracted as billing despite explicit negation in source text.")

        # Sarcasm check: ironic praise mixed with failure words
        sarcasm_cues = ["love it", "oh great", "another billing", "surprise"]
        if any(c in text_lower for c in sarcasm_cues) and ("crash" in text_lower or "surprise" in text_lower or "bug" in text_lower):
            warnings.append("Potential sarcasm detected; verify urgency and intent manually.")

        # Summary grounding check
        summary = result.get("summary", "")
        if summary:
            first_words = [w for w in re.findall(r"\w+", summary.lower())[:3] if len(w) > 2]
            if first_words and not any(w in text_lower for w in first_words):
                warnings.append("Summary may not be grounded in source text.")

        # Entity hallucination check
        entities = result.get("entities", [])
        for ent in entities:
            ent_name = ent.get("name", "") if isinstance(ent, dict) else getattr(ent, "name", "")
            if ent_name and ent_name.lower() not in text_lower:
                warnings.append(f"Entity '{ent_name}' not found in source text (possible hallucination).")

    # Invoice-specific checks
    if "invoice_number" in result or "vendor" in result:
        if result.get("amount") == 0.0:
            warnings.append("Amount not found in text, defaulted to 0 — do not trust downstream.")
        if result.get("due_date") == "not specified" and conf > 0.8:
            warnings.append("Confident despite missing due date.")

    return warnings
