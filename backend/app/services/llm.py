"""Google Generative Language client. One instance per model, each rate-limited.

Model-agnostic: it speaks the plain generateContent REST API, so the same class
serves Gemini Flash, Flash Lite and Gemma. Which model a given call uses is a
config decision (see `Settings.limits_for`), not a code one.

WHY ONE CLIENT PER MODEL
Free-tier request quota is PER MODEL -- 5/min for the full Flash models, 15 for
Flash Lite, 30 for Gemma -- so each needs its own token bucket. Sharing one
would throttle every call on the strictest limit.

STRUCTURED OUTPUT
Every structured step goes through generationConfig.responseSchema. Never use
responseMimeType without a schema: the model emits its own reasoning trace
instead of the object.

Verified live on this key, and the reason the model split looks the way it does:
gemini-3.5-flash VIOLATES a responseSchema (it returned "Here is the JSON
requested:" as prose), while gemini-3.8-flash and gemini-3.5-flash-lite honour
it exactly. Every Gemini Flash model here reports a 1,048,576-token input
window and supports native functionDeclarations -- unlike Gemma, which narrates
tool use as prose, which is why schemas remain the mechanism.

Shares no quota with embeddings, which have their own limiter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.services import tracing
from app.services.limiter import RateLimiter, estimate_tokens

log = structlog.get_logger()

GENAI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    """A model call failed.

    Carries the HTTP status when there was one, because the CALLER often has a
    better remedy than a retry: a pool can move to a different model on 429,
    where backing off on the exhausted one just waits. Without the status that
    decision would have to be made by matching on the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        # Seconds, parsed from the 429's RetryInfo when Google supplies one.
        # Better than a guessed backoff: it is the server saying when the
        # budget actually refills.
        self.retry_after = retry_after


def _retry_after(response) -> float | None:
    """`retryDelay` from a 429 body, in seconds."""
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except (ValueError, TypeError):
        pass
    return None


class GenAIClient:
    """A Google Generative Language client for ONE model, with its own limiter.

    Was `GemmaClient`. Renamed because it now serves Gemini for everything a
    user sees and Gemma only for query rewriting -- a class named after one
    model while serving three is a comment that lies.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
    ) -> None:
        s = get_settings()
        if not s.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required")

        self._model = (model or s.llm_model).removeprefix("models/")
        self._client = httpx.AsyncClient(
            base_url=GENAI_BASE,
            timeout=httpx.Timeout(180.0),
            headers={"x-goog-api-key": s.google_api_key},
        )
        self._limiter = RateLimiter(
            requests_per_minute=requests_per_minute or s.llm_requests_per_minute,
            tokens_per_minute=tokens_per_minute or s.llm_tokens_per_minute,
            # Named per model so limiter log lines say WHICH budget throttled.
            name=f"llm:{self._model}",
        )

    @property
    def model(self) -> str:
        """The model id this client calls, without the `models/` prefix."""
        return self._model

    def wait_estimate(self, tokens: int) -> float:
        """Seconds before this client could serve a request of `tokens`.

        Consumes nothing -- a pool has to be able to compare several members
        without spending budget on the ones it does not choose.
        """
        return self._limiter.peek(tokens)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.9,
        max_attempts: int = 4,
    ) -> str:
        """One completion. Returns raw text (JSON text when `schema` is given)."""
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                # Repetition loops are the characteristic failure of the
                # smaller models here -- Gemma once emitted "way's actually" a
                # dozen times until it hit the token cap, producing truncated
                # JSON, and later several hundred "the-the". The API exposes no
                # repetition penalty, so nucleus sampling plus a low
                # temperature is the available defence, and
                # `strip_degeneration` below is the net underneath it.
                "topP": top_p,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = schema

        # Budget both directions: the prompt we send and the tokens we allow
        # back, since TPM counts both.
        cost = estimate_tokens(prompt) + estimate_tokens(system or "") + max_output_tokens

        # Traced as a GENERATION, not a plain span: that observation type is
        # what gets Langfuse's token and cost accounting, model breakdown and
        # latency-per-model charts. The LangChain callback handler cannot see
        # this call -- it goes over raw httpx, not through LangChain -- which is
        # exactly why it is instrumented by hand.
        with tracing.observe(
            "genai.generate",
            as_type="generation",
            input={"system": system, "prompt": prompt},
            model=self._model,
            model_parameters={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                # Whether the call was schema-constrained. Worth a dimension:
                # the unstructured calls are the ones that degenerate.
                "structured": bool(schema),
            },
        ) as span:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                # Timed here rather than by making `acquire` return a duration:
                # the limiter's contract stays "block until it fits", and the
                # measurement is the caller's concern.
                before = time.perf_counter()
                await self._limiter.acquire(cost)
                waited = time.perf_counter() - before
                try:
                    resp = await self._client.post(
                        f"/models/{self._model}:generateContent", json=body
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    text = _extract_text(payload)
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code not in (429, 500, 502, 503):
                        tracing.update(
                            span,
                            level="ERROR",
                            status_message=f"{exc.response.status_code}",
                        )
                        raise LLMError(
                            f"{exc.response.status_code}: {exc.response.text[:400]}",
                            status=exc.response.status_code,
                        ) from exc
                    # LAST ATTEMPT: raise with the status rather than sleeping
                    # into a retry that will not happen. A pool calls this with
                    # max_attempts=1 precisely so it can move to a model that
                    # still has budget -- backing off here would spend the time
                    # the pool exists to avoid.
                    if attempt == max_attempts - 1:
                        tracing.update(
                            span,
                            level="ERROR",
                            status_message=f"{exc.response.status_code}",
                        )
                        raise LLMError(
                            f"{exc.response.status_code}: {exc.response.text[:200]}",
                            status=exc.response.status_code,
                            retry_after=_retry_after(exc.response),
                        ) from exc
                    backoff = 2**attempt * 5
                    log.warning(
                        "llm_retry",
                        status=exc.response.status_code,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                except LLMError as exc:
                    # A 200 with no usable text: safety block, recitation, or
                    # MAX_TOKENS with empty parts. The finish reason is the
                    # actionable part, so it goes on the span.
                    tracing.update(span, level="ERROR", status_message=str(exc)[:200])
                    raise

                usage = payload.get("usageMetadata") or {}
                tracing.update(
                    span,
                    output=text,
                    # Real counts from the provider where available, falling
                    # back to the estimate the limiter used. Cost analytics is
                    # only as good as these numbers.
                    usage_details={
                        "input": usage.get("promptTokenCount", estimate_tokens(prompt)),
                        "output": usage.get(
                            "candidatesTokenCount", estimate_tokens(text)
                        ),
                    },
                    metadata={
                        "attempts": attempt + 1,
                        # Seconds spent waiting on this app's own rate limiter.
                        # Distinguishes "the model was slow" from "we throttled
                        # ourselves", which look identical in wall-clock latency
                        # and have completely different fixes.
                        "throttled_seconds": round(waited, 2),
                        # RECITATION means the model was reproducing memorised
                        # text -- in RAG that is a GROUNDING failure, not a
                        # token-limit problem, and retrying with more tokens is
                        # the wrong fix. Filterable here.
                        "finish_reason": (payload.get("candidates") or [{}])[0].get(
                            "finishReason"
                        ),
                    },
                )
                return text

            tracing.update(span, level="ERROR", status_message="retries exhausted")
            raise LLMError(f"LLM failed after retries: {last_error}")

    async def generate_tools(
        self,
        contents: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """One round of native tool calling.

        Returns `(function_calls, text, raw_model_content)`:

        * `function_calls` -- what the model wants executed, each
          `{"name": str, "args": dict}`. Empty when it is done and answering.
        * `text` -- any prose parts, which is the final answer once it stops
          calling tools.
        * `raw_model_content` -- the model's own `content` block, which the
          CALLER MUST append to `contents` before adding tool results. The
          conversation has to contain the request as well as the response or
          the model loses track of what it asked for.

        Native functionDeclarations, not a text protocol. The original ReAct
        parsed `Thought:/Action:` out of a completion, which is brittle and the
        source of endless "could not parse LLM output" errors. Gemini returns a
        structured `functionCall` part instead -- verified live on this key.
        Gemma cannot do this at all: given tool declarations it narrates what
        it would do in prose, which is why this exists only now.
        """
        body: dict[str, Any] = {
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        # Rough: the whole conversation grows each round, so cost is charged on
        # the current transcript rather than one prompt.
        cost = sum(
            estimate_tokens(part.get("text", ""))
            for message in contents
            for part in message.get("parts", [])
        ) + estimate_tokens(system or "") + max_output_tokens

        with tracing.observe(
            "genai.tools",
            as_type="generation",
            input=contents[-1] if contents else None,
            model=self._model,
            model_parameters={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "n_tools": sum(len(t.get("functionDeclarations", [])) for t in tools),
            },
        ) as span:
            await self._limiter.acquire(cost)
            try:
                resp = await self._client.post(
                    f"/models/{self._model}:generateContent", json=body
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                tracing.update(
                    span, level="ERROR", status_message=str(exc.response.status_code)
                )
                raise LLMError(
                    f"{exc.response.status_code}: {exc.response.text[:400]}"
                ) from exc

            payload = resp.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                feedback = payload.get("promptFeedback", {})
                tracing.update(span, level="ERROR", status_message="no candidates")
                raise LLMError(f"no candidates returned (promptFeedback={feedback})")

            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            text = "".join(p.get("text", "") for p in parts).strip()

            usage = payload.get("usageMetadata") or {}
            tracing.update(
                span,
                output={"calls": calls, "text": text},
                usage_details={
                    "input": usage.get("promptTokenCount", 0),
                    "output": usage.get("candidatesTokenCount", 0),
                },
                metadata={
                    "n_calls": len(calls),
                    "finish_reason": candidates[0].get("finishReason"),
                },
            )
            return calls, strip_degeneration(text), content

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> Any:
        """Structured output. This is our substitute for tool-calling."""
        raw = await self.generate(
            prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Even with a schema, a truncated response (hit maxOutputTokens
            # mid-object) is invalid JSON. Surface it clearly rather than
            # letting a caller crash on a missing key.
            raise LLMError(f"model returned invalid JSON: {raw[:400]}") from exc


def extract_string_list(raw: str, key: str) -> list[str]:
    """Pull a list of strings out of possibly-truncated JSON.

    Gemma sometimes produces a valid array and then degenerates in a LATER
    field, truncating the response and invalidating the whole document. The
    array itself is fine, so discarding everything would throw away good work
    -- which is exactly the bug this fixes.

    Tries a strict parse first; falls back to scanning the named array for
    complete "..." literals (the truncated final one has no closing quote and
    simply isn't matched).
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            items = data.get(key, [])
            return [s.strip() for s in items if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)(?:\]|$)', raw, re.DOTALL)
    if not match:
        return []

    found = [
        m.group(1).replace('\\"', '"').replace("\\'", "'").strip()
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    ]
    return [s for s in found if s and not is_repetitive(s)]


def is_repetitive(text: str) -> bool:
    """True if a few distinct words fill a long string -- a loop artefact."""
    words = text.lower().split()
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < 0.4


# A short unit repeated consecutively at least five times. Bounded on both
# sides deliberately: `.{2,40}` keeps the backreference search cheap, and
# requiring five repeats avoids cutting legitimate prose ("very, very good",
# a row of dashes, "ha ha ha").
_MAX_UNIT = 40
_DEGENERATION = re.compile(rf"(.{{2,{_MAX_UNIT}}}?)\1{{4,}}", re.DOTALL)

# Only the tail is scanned. A repetition loop runs until the token limit, so it
# is always at the END -- and capping the scan keeps the backreference search
# from getting expensive on a long answer.
_DEGENERATION_SCAN = 2000


def strip_degeneration(text: str) -> str:
    """Cut a response at the point it started repeating itself.

    Gemma's characteristic failure is a repetition loop: it emits a plausible
    sentence, then loops a fragment until it hits max_output_tokens. Measured
    example, from a real turn:

        "...the provided sources do not contain information regarding which
         specific pyramid you are asking about. [No source provided for this
         clarification/refusal/unanswered part of the
         question/question/question/question/question/..."

    The first sentence is a perfectly good answer. Everything from
    "question/question" on is noise, and because the loop ran until the token
    cap it also truncated the JSON -- which is what turned a usable answer into
    a 502.

    Returns the text up to the loop. Never raises, and returns the input
    unchanged when nothing repeats.
    """
    if not text:
        return text

    head, tail = text[:-_DEGENERATION_SCAN], text[-_DEGENERATION_SCAN:]

    # Only a run that reaches the END counts.
    #
    # The loop runs until the token cap, so it is always the last thing in the
    # response. Taking the earliest match instead cut legitimate content that
    # merely contains a repeated pattern -- a rule of dashes, a run of spaces,
    # a table separator. Requiring the run to reach the end removes that whole
    # class of false positive. The trailing slack absorbs a final unit that
    # truncation cut in half.
    unit = None
    start = None
    for match in _DEGENERATION.finditer(tail):
        if match.end() >= len(tail) - _MAX_UNIT:
            unit = match.group(1)
            start = match.start()
            break
    if unit is None or start is None:
        return text

    # EXTEND BACKWARDS past the scan window.
    #
    # The window bounds the regex, not the run. A loop longer than the window
    # begins before it, and cutting at the window boundary left the earlier
    # part in place: measured 3364 chars in, 1363 out, with ~1200 characters of
    # "the-the the-the ..." still in the answer the user read.
    #
    # The unit the regex found may be a rotation of the true period, which does
    # not matter -- a periodic string repeats under any rotation of its period.
    cut_at = len(head) + start
    step = len(unit)
    while cut_at - step >= 0 and text[cut_at - step : cut_at] == unit:
        cut_at -= step

    cut = text[:cut_at].rstrip()
    # A loop starting in the first few characters means there is no real answer
    # to salvage; returning "" lets the caller say so honestly rather than
    # showing a fragment.
    return cut if len(cut) >= 20 else ""


def _unescape_newlines(value: str) -> str:
    r"""Turn literal ``\n`` into real newlines -- but only when there are none.

    A model asked to put line breaks in a JSON string field sometimes escapes
    the BACKSLASH instead of the newline, emitting ``"\\n"`` in the JSON. That
    decodes to the two visible characters ``\`` and ``n``, which reach the user
    as "records [5] .\n\nRegarding the technical operations..." -- measured,
    from a real turn.

    THE GUARD IS THE WHOLE DESIGN. Only text containing no real newline at all
    is touched. A model that produced genuine line breaks was clearly capable
    of it, so a backslash-n still sitting in that text is deliberate content --
    a regex in a code block, a Windows path, an explanation of escaping itself
    -- and rewriting it would corrupt the answer. Text with zero real breaks
    and several literal ones can only be the failure above.
    """
    if "\n" in value or "\\n" not in value:
        return value
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def extract_string(raw: str, key: str) -> str:
    """Pull a possibly-truncated string value out of JSON.

    The string counterpart to `extract_string_list`, and it exists for the same
    reason: a response that degenerated in one field still usually carries a
    complete, useful value in another, and a strict parse throws all of it away.

    Three attempts, weakest last:
      1. parse the whole document
      2. find a properly closed "key": "..." pair
      3. take everything after the opening quote -- the truncated case
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get(key), str):
            return strip_degeneration(_unescape_newlines(data[key]).strip())
    except json.JSONDecodeError:
        pass

    quoted = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(quoted, raw, re.DOTALL)
    if match is None:
        # Unterminated: the closing quote was never emitted.
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"(.*)$', raw, re.DOTALL)
    if match is None:
        return ""

    value = match.group(1)
    try:
        # Round-trip through the JSON decoder so escapes are handled properly
        # rather than by hand. A trailing lone backslash makes this fail, hence
        # the fallback.
        value = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        value = value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    return strip_degeneration(_unescape_newlines(value).strip())


def extract_object_list(raw: str, key: str) -> list[dict]:
    """Pull a list of OBJECTS out of possibly-truncated JSON.

    The object-array counterpart to `extract_string_list`, and it exists for a
    measured failure: the clarify node asks for 2-4 `{label, description}`
    options, Gemma degenerated inside the THIRD description, and the truncated
    document took `ambiguous: true` and two complete, usable options down with
    it. The turn then ran without pausing -- so a repetition loop in a field
    nobody reads silently disabled the feature.

    Scans the named array brace-by-brace and keeps only objects that closed.
    The truncated final one simply never balances, so it is skipped.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            items = data.get(key, [])
            return [x for x in items if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    start = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not start:
        return []

    out: list[dict] = []
    depth = 0
    begin = -1
    in_string = False
    escaped = False
    for i in range(start.end(), len(raw)):
        ch = raw[i]
        # String-aware, so a brace inside a description does not shift depth.
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and begin >= 0:
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(raw[begin : i + 1])
                    if isinstance(parsed, dict):
                        out.append(parsed)
                begin = -1
        elif ch == "]" and depth == 0:
            break
    return out


def extract_bool(raw: str, key: str, *, default: bool = False) -> bool:
    """Pull a boolean out of possibly-truncated JSON."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get(key), bool):
            return data[key]
    except json.JSONDecodeError:
        pass

    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return default


def extract_int_list(raw: str, key: str) -> list[int]:
    """Pull a list of integers (citation numbers) out of possibly-broken JSON."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [n for n in data.get(key, []) if isinstance(n, int)]
    except json.JSONDecodeError:
        pass

    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[([^\]]*)', raw, re.DOTALL)
    if not match:
        return []
    return [int(n) for n in re.findall(r"\d+", match.group(1))]


# Why a 200 can carry no usable text. The old error told the operator to raise
# max_output_tokens regardless of cause, which is only ever right for
# MAX_TOKENS -- and actively misleading for RECITATION, where more tokens
# cannot help.
_EMPTY_REASONS = {
    "MAX_TOKENS": (
        "the model hit its output limit before emitting anything usable. "
        "Raise max_output_tokens for this call."
    ),
    # Google stops generation when output starts reproducing memorised training
    # data. In this app it means the question was NOT answerable from the
    # retrieved passages, so the model fell back on world knowledge and began
    # reciting a remembered fact. The trigger is a grounding failure, not a
    # configuration problem.
    "RECITATION": (
        "the model began reproducing memorised training text and Google "
        "stopped it. This usually means the question is not answerable from "
        "the retrieved passages, so the model fell back on world knowledge. "
        "Raising max_output_tokens will not help."
    ),
    "SAFETY": "the response was blocked by a safety filter.",
    "PROHIBITED_CONTENT": "the response was blocked as prohibited content.",
    "OTHER": "generation stopped for an unspecified reason.",
}


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull text out of a generateContent response.

    Defensive because there are several ways to get a 200 with no usable text:
    a safety block, a recitation stop, or finishReason=MAX_TOKENS with an empty
    parts list. The finish reason decides what the operator should actually do,
    so it is named in the message rather than guessed at.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback", {})
        raise LLMError(f"no candidates returned (promptFeedback={feedback})")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)

    if not text.strip():
        reason = candidate.get("finishReason") or "UNKNOWN"
        explanation = _EMPTY_REASONS.get(
            reason, "no text was returned and the finish reason is unrecognised."
        )
        raise LLMError(f"empty response ({reason}): {explanation}")
    return text


# One client PER MODEL, not one per process.
#
# Each entry owns its own RateLimiter, and that is the whole point: free-tier
# quotas are per model and have opposite shapes. Gemma 4 allows 30 requests and
# 14,400/day but only 16K tokens/minute; Gemini Flash Lite allows 250K
# tokens/minute but only 500 requests/day. A shared limiter would throttle both
# on whichever numbers it happened to be configured with, and silently waste
# most of the combined budget.
#
# Routing follows the shapes: small frequent calls (plan, query expansion) to
# Gemma, large context-carrying calls (draft, critique, judging) to Flash Lite.
_clients: dict[str, GenAIClient] = {}


def get_llm(model: str | None = None) -> GenAIClient:
    """Client for `model`, defaulting to `settings.llm_model`.

    Cached per model name so the limiter state persists across calls -- a fresh
    client per request would start with a full token bucket and defeat rate
    limiting entirely.
    """
    settings = get_settings()
    name = model or settings.llm_model
    if name not in _clients:
        rpm, tpm = settings.limits_for(name)
        _clients[name] = GenAIClient(model=name, requests_per_minute=rpm, tokens_per_minute=tpm)
        log.info("llm_ready", model=name, rpm=rpm, tpm=tpm)
    return _clients[name]


async def close_llm() -> None:
    for client in _clients.values():
        await client.aclose()
    _clients.clear()
