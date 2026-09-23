You are modifying an existing LLM client project.

## Goal

Improve the existing **Acceptance Test / Fault Injection Simulation** tab so that the failure-handling behavior is visually obvious during a 5-minute technical demo.

**Do NOT redesign the entire application.**
Keep the existing functionality, API behavior, retry logic, logs, and styling direction. Only enhance the Acceptance Test UI/UX and connect the visualization to the existing fault-injection results.

---

# Existing Acceptance Test

The application already has a Fault Injection Simulation section with scenarios such as:

* 429 Rate Limit
* 500 Provider Error
* Timeout
* 400 Bad Request
* Malformed Output

It already produces live attempt logs similar to:

```text
📋 Live Attempt Log

Attempt 1
Rate Limited (429)

0 ms elapsed

↳ Backing off for 0.21s
   (exponential backoff + full jitter)

rate limit exceeded

Attempt 2
Rate Limited (429)

205 ms elapsed

↳ Backing off for 1.23s
   (exponential backoff + full jitter)

rate limit exceeded

Attempt 3
Rate Limited (429)

1438 ms elapsed

↳ Maximum retry attempts reached
   No further retries permitted

rate limit exceeded

🛑 Client Raised LLMRateLimitError

Detail: rate limit exceeded

✓ Failure handled safely according to the configured
  retry policy.
```

Preserve this existing log.

---

# New UI Requirement

Add a visual **Failure Flow / Execution Timeline** above or beside the existing logs.

The purpose is to allow a technical evaluator to understand the behavior immediately without reading every log line.

The visual flow should clearly communicate:

```text
Request
   ↓
Provider Response / Failure
   ↓
Retry Decision
   ↓
Backoff
   ↓
Retry
   ↓
Success OR Final Failure
```

The visualization must update dynamically based on the selected fault-injection scenario.

---

# 1. Main Acceptance Test Layout

Create a clean layout approximately like:

```text
┌─────────────────────────────────────────────────────────────┐
│              FAULT INJECTION / ACCEPTANCE TEST              │
│                                                             │
│ [429 Rate Limit] [500 Provider] [Timeout]                   │
│ [400 Bad Request] [Malformed Output]                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                 FAILURE EXECUTION FLOW                     │
│                                                             │
│       ┌──────────────┐                                     │
│       │    REQUEST   │                                     │
│       └──────┬───────┘                                     │
│              ↓                                             │
│       ┌──────────────┐                                     │
│       │ HTTP 429     │                                     │
│       │ RATE LIMIT   │                                     │
│       └──────┬───────┘                                     │
│              ↓                                             │
│       ↙ BACKOFF 0.21s ↘                                    │
│              ↓                                             │
│       ┌──────────────┐                                     │
│       │   RETRY #2   │                                     │
│       └──────┬───────┘                                     │
│              ↓                                             │
│       ┌──────────────┐                                     │
│       │ HTTP 429     │                                     │
│       └──────┬───────┘                                     │
│              ↓                                             │
│       ↙ BACKOFF 1.23s ↘                                    │
│              ↓                                             │
│       ┌──────────────┐                                     │
│       │   RETRY #3   │                                     │
│       └──────┬───────┘                                     │
│              ↓                                             │
│       ┌──────────────┐                                     │
│       │ MAX RETRIES  │                                     │
│       └──────────────┘                                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 📋 Live Attempt Log                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The exact layout can be adapted to the existing application's design.

---

# 2. Dynamic Animation

When the user starts a fault-injection test, do not display the entire flow instantly.

Animate the execution sequentially.

Example for 429:

```text
REQUEST
   ↓
429 RATE LIMIT
   ↓
BACKOFF 0.21s
   ↓
RETRY #2
   ↓
429 RATE LIMIT
   ↓
BACKOFF 1.23s
   ↓
RETRY #3
   ↓
429 RATE LIMIT
   ↓
MAX RETRIES REACHED
```

Each node should appear/highlight as the actual simulated attempt occurs.

Use subtle transitions such as:

* fade-in
* slide-in
* pulse on active node
* progress indicator
* animated connector/arrow

Do not use excessive visual effects.

The animation should feel like a technical execution trace.

---

# 3. 429 Rate Limit Visualization

For:

```text
429 Rate Limit
```

show:

```text
🟢 REQUEST
   ↓
🔴 429 RATE LIMIT
   ↓
🟡 BACKOFF
   0.21s
   ↓
🔵 RETRY #2
   ↓
🔴 429 RATE LIMIT
   ↓
🟡 BACKOFF
   1.23s
   ↓
🔵 RETRY #3
   ↓
🔴 429 RATE LIMIT
   ↓
🛑 MAX RETRIES
```

Clearly display:

* attempt number
* HTTP status
* backoff duration
* retry decision
* final result

The backoff values must come from the actual simulation/log data if available.

Do not invent values purely for the animation.

---

# 4. 500 Provider Error Visualization

For a simulated provider failure:

```text
REQUEST
   ↓
HTTP 500
   ↓
RETRY
   ↓
HTTP 500
   ↓
RETRY
   ↓
HTTP 200
   ↓
SUCCESS
```

If the existing simulation eventually succeeds, show:

```text
🟢 RECOVERED
```

If it exhausts retries, show:

```text
🛑 MAX RETRIES
```

The visualization must reflect the actual result.

---

# 5. Timeout Visualization

For timeout:

```text
REQUEST
   ↓
⏱ WAITING
   ↓
2.0s TIMEOUT
   ↓
RETRY
   ↓
⏱ WAITING
   ↓
2.0s TIMEOUT
   ↓
RETRY
   ↓
...
   ↓
🛑 MAX RETRIES
```

Add a small visual timer/progress animation while waiting.

Display the configured timeout duration.

Do not block the UI while the timeout simulation runs.

---

# 6. 400 Bad Request Visualization

This case is especially important because it demonstrates **fast failure**.

Show:

```text
REQUEST
   ↓
HTTP 400
BAD REQUEST
   ↓
🛑 NO RETRY
   ↓
LLMBadRequestError
```

Make it visually obvious that there is **no backoff and no second attempt**.

The UI should communicate:

```text
Retryable: NO
Reason: Permanent/client-side request error
```

Do not call this "retry limit reached".

This is an immediate fast-fail decision.

---

# 7. Malformed Output Visualization

For malformed structured output:

```text
LLM REQUEST
   ↓
HTTP 200
   ↓
JSON RESPONSE
   ↓
PYDANTIC VALIDATION
   ↓
❌ VALIDATION FAILED
   ↓
🛑 NO RETRY
   ↓
LLMOutputValidationError
```

This should visually demonstrate an important concept:

**HTTP 200 does not necessarily mean a valid application response.**

Show:

```text
HTTP Status: 200 OK
Schema Validation: FAILED
Retry: NO
```

---

# 8. Status/Decision Panel

Add a compact panel showing the current policy decision.

Example:

```text
┌───────────────────────────────┐
│ CURRENT DECISION              │
├───────────────────────────────┤
│ Error: HTTP 429               │
│ Retryable: YES                │
│ Attempt: 2 / 3                │
│ Backoff: 1.23s                │
│ Strategy: Exponential + Jitter│
│ Status: RETRYING              │
└───────────────────────────────┘
```

For 400:

```text
┌───────────────────────────────┐
│ CURRENT DECISION              │
├───────────────────────────────┤
│ Error: HTTP 400               │
│ Retryable: NO                 │
│ Attempt: 1 / 3                │
│ Backoff: —                    │
│ Strategy: Fast Fail           │
│ Status: TERMINATED            │
└───────────────────────────────┘
```

---

# 9. Final Result Banner

At the end of every simulation, show a clear result.

Examples:

### Successful recovery

```text
✓ RECOVERED

Provider failure occurred,
retry policy was applied,
and the request eventually succeeded.

Attempts: 3
Final Status: 200
```

### Retry exhaustion

```text
🛑 RETRY LIMIT REACHED

The client stopped after the configured
maximum number of attempts.

Attempts: 3
Final Error: LLMRateLimitError
```

### Fast failure

```text
🛑 FAST FAIL

This error is not retryable.

Attempts: 1
Final Error: LLMBadRequestError
```

### Validation failure

```text
❌ OUTPUT VALIDATION FAILED

Provider returned HTTP 200,
but the response did not satisfy
the Pydantic schema.

Retry: NO
```

---

# 10. Preserve Existing Logs

Do NOT replace the existing Live Attempt Log.

The final screen should provide both:

```text
Visual Execution Flow
        +
Live Technical Logs
```

The visual flow is for quick understanding.

The logs are the technical evidence.

---

# 11. Clear Logs

Keep the existing:

```text
Clear Logs
```

button.

When clicked:

* clear attempt logs
* clear execution flow
* reset status panel
* reset final result
* return UI to idle state

---

# 12. Important Engineering Constraint

Do not duplicate the retry logic in the frontend purely to create the animation.

The visualization should consume the **actual events/results generated by the existing fault-injection/client implementation**.

Ideally expose events such as:

```text
request_started
provider_error
retry_scheduled
backoff_started
retry_started
validation_failed
request_succeeded
retry_exhausted
fast_failed
```

Then map those events to visual nodes.

This ensures the demo cannot visually claim something different from what the client actually did.

---

# 13. Keep the Demo Professional

The visual design should be:

* clean
* technical
* minimal
* easy to understand
* responsive
* suitable for a 5-minute engineering demo

Avoid:

* excessive animations
* sound effects
* unnecessary gradients
* huge decorative graphics
* fake metrics
* fake retry behavior
* duplicating backend logic in frontend

The visual should explain the engineering behavior, not distract from it.

---

# 14. Acceptance Test Scenarios

Ensure the UI clearly supports these five scenarios:

| Scenario           | Expected visual behavior                                |
| ------------------ | ------------------------------------------------------- |
| 429 Rate Limit     | Retry → backoff → retry → eventually success/exhaustion |
| 500 Provider Error | Retry → backoff → retry → recovery/exhaustion           |
| Timeout            | Timeout → retry → timeout → retry → exhaustion          |
| 400 Bad Request    | Immediate failure → NO retry                            |
| Malformed Output   | HTTP 200 → Pydantic validation failure → NO retry       |

---

# 15. Final Demo Objective

When I click a fault-injection scenario, an evaluator should immediately understand:

1. What failure occurred?
2. Was it retryable?
3. Why was it retried or not retried?
4. How many attempts occurred?
5. What backoff was applied?
6. Did the request recover?
7. If it failed, why did the client stop?
8. What exception was ultimately raised?

The visual flow and existing logs together must answer all eight questions.

Before finishing, verify that the existing live LLM prompt/Speed Insights tab remains unchanged and functional.
