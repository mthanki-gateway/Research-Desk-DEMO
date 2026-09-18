from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Model profiles
#
# One switch that moves every model AND every budget that depends on it.
#
# WHY THIS EXISTS. The richer retrieval/agent design costs 9-10 model calls per
# turn (multi-query, rerank, several agent hops, draft, critique). The answer
# model allows FIVE REQUESTS PER MINUTE, so exercising that design on Gemini
# means roughly one question every two minutes -- unusable for development.
# Gemma allows 30 rpm and 14,400/day, which is the only budget on this key that
# can actually run the loop repeatedly.
#
# The trade is real and the profile encodes both halves of it. Gemma is faster
# and far more forgiving of quota, but it has a 16K tokens/MINUTE ceiling and
# NO native tool calling -- so the profile also shrinks history, shortens the
# draft, and turns ReAct off, because those are not independent choices.
#
# A profile only supplies DEFAULTS. An explicit env var always wins, so
# LLM_MODEL=... in .env still overrides whatever the profile would have set.
# --------------------------------------------------------------------------

_GEMMA = "models/gemma-4-26b-a4b-it"

MODEL_PROFILES: dict[str, dict[str, object]] = {
    # The production shape. Three models split by per-model request quota.
    "gemini": {
        "llm_model": "models/gemini-3.5-flash-lite",
        "llm_requests_per_minute": 15,
        "llm_tokens_per_minute": 250_000,
        # THE ANSWER MODEL IS ON THE 15 RPM TIER, not the 5 rpm one.
        #
        # It was gemini-3.6-flash, which is the stronger model and allows FIVE
        # REQUESTS PER MINUTE. A turn spends 1-2 of them on the draft, so two
        # questions in quick succession -- or one question plus a critique
        # retry -- exhausted the budget and stalled in backoff. Measured
        # repeatedly during development, including three of four evaluation
        # questions failing with 429 in a single run.
        #
        # Verified rather than assumed: seven requests fired inside one minute
        # returned 200 with zero 429s, which is what distinguishes this tier
        # from the 5 rpm one. gemini-3.1-flash-lite was 503 (unavailable) and
        # gemini-2.5-flash-lite flaked once in the same burst, so 3.5 is the
        # pick. `gemini-flash-lite-latest` also passed and is deliberately NOT
        # used: a moving alias cannot be the model an evaluation is reported
        # against.
        "answer_model": "models/gemini-3.5-flash-lite",
        "answer_requests_per_minute": 15,
        "answer_tokens_per_minute": 250_000,
        # Now the STRONGER model, which is the right way round and was not
        # previously true. The critic must be at least as strong as the
        # generator or it rubber-stamps, and the old split had flash-lite
        # grading 3.6-flash -- the weaker model judging the stronger one.
        #
        # Its 5 rpm is affordable HERE in a way it was not for answering:
        # judging is an offline batch job with answers already cached, so it
        # costs wall-clock on an evaluation run rather than latency on a
        # request. For fast iteration set JUDGE_MODEL=models/gemini-flash-lite-
        # latest, at the cost of a judge no stronger than the generator.
        "judge_model": "models/gemini-3.6-flash",
        "judge_requests_per_minute": 5,
        "judge_tokens_per_minute": 250_000,
        # The pool belongs to the PROFILE, not to a standalone default.
        #
        # Left as a plain default it would follow the gemma profile across and
        # put a Gemini model in a Gemma pool, which `ModelPool` refuses at
        # construction -- so switching profile would raise instead of switching.
        # Pools and models have to move together for the same reason the token
        # budgets do.
        "answer_model_pool": (
            "models/gemini-3.1-flash-lite,"
            "models/gemini-3.1-flash-lite-preview,"
            "models/gemini-2.5-flash-lite"
        ),
        # The workhorse makes the MOST calls per turn -- plan, clarify,
        # critique, rerank -- so it pools the same set. They share limiters
        # with the answer role, which is correct: quota is per model, and a
        # bursty role borrowing from a quiet one is the point.
        "llm_model_pool": (
            "models/gemini-3.1-flash-lite,"
            "models/gemini-3.1-flash-lite-preview,"
            "models/gemini-2.5-flash-lite"
        ),
        "rewriter_model": _GEMMA,
        "history_full": True,
        "history_max_tokens": 200_000,
        "verbatim_messages": 40,
        "draft_max_output_tokens": 2000,
        "react_default": True,
        "react_max_rounds": 6,
        "react_max_calls_per_round": 4,
        "agent_max_iterations": 2,
    },
    # Development. ONE model for every role, because the quota is per model and
    # Gemma's is the only one large enough to run this loop on repeat.
    "gemma": {
        "llm_model": _GEMMA,
        "llm_requests_per_minute": 30,
        # 16K tokens/MINUTE is the real ceiling here, and it is what every
        # budget below is derived from. A single turn at 3-5 calls has roughly
        # 3-5K tokens per call to spend on everything: history, retrieved
        # passages, and the output.
        "llm_tokens_per_minute": 16_000,
        # Same model for the answer. Not a compromise on quality so much as an
        # acknowledgement that there is no second Gemma to promote to.
        "answer_model": _GEMMA,
        "answer_requests_per_minute": 30,
        "answer_tokens_per_minute": 16_000,
        # Stays on Flash Lite: stronger than Gemma (so it does not rubber-stamp)
        # and 15 rpm rather than 5, which matters because this profile exists to
        # make evaluation runs affordable in the first place.
        "judge_model": "models/gemini-3.5-flash-lite",
        "judge_requests_per_minute": 15,
        "judge_tokens_per_minute": 250_000,
        # NO POOL. gemma-4-31b-it is the only other Gemma on this key and it
        # returns 500 on every call (measured). A pool member that always
        # fails is worse than none: it consumes a pick and then fails the call
        # it was picked for.
        "answer_model_pool": "",
        "llm_model_pool": "",
        "rewriter_model": _GEMMA,
        # The whole transcript does not fit in a 16K/minute budget -- it was
        # the arrival of Gemini's 250K that made HISTORY_FULL possible at all.
        # Back to the rolling summary plus a short verbatim tail.
        "history_full": False,
        "history_max_tokens": 6_000,
        "verbatim_messages": 6,
        # 2000 output tokens is a third of a minute's entire budget on Gemma,
        # and it degenerates into repetition loops long before reaching it.
        "draft_max_output_tokens": 900,
        # OFF, and this one is not a budget decision -- it is a capability one.
        # Gemma emits no functionCall parts; given tool declarations it narrates
        # what it would do in prose, which the ReAct loop reads as "no tools
        # requested" and exits on round 1. See `supports_tool_calling`.
        "react_default": False,
        "react_max_rounds": 3,
        "react_max_calls_per_round": 2,
        # One critique cycle, not two. Each costs ~2 calls and Gemma's token
        # ceiling is the binding constraint, not its request count.
        "agent_max_iterations": 1,
    },
}

# Profiles whose model emits native functionCall parts. Verified live: every
# Gemini Flash model on this key does; Gemma does not, at all.
_TOOL_CALLING_PROFILES = frozenset({"gemini"})

# Requests/minute per model, MEASURED rather than assumed.
#
# The API does not report rate limits -- `GET /models/{name}` returns token
# sizes only -- so these come from `python -m app.scripts.probe_limits`, which
# trips each model's limit once and reads `quotaValue` out of the 429 body.
#
# Re-run it when models change. The numbers are per key and Google moves them.
#
# Omitted deliberately, with the reason, so nobody re-adds them hopefully:
#   gemini-3.8-flash, gemini-3.7-flash   503 -- published but not serving
#   gemini-3.1-pro-preview, -pro-latest  429 with NULL quota -- 0/0, unusable
#   gemini-2.5-pro                       404 -- withdrawn for new keys
#   gemma-4-31b-it                       500 -- internal error on every call
#   *-latest aliases                     work, but a moving target cannot be
#                                        pinned, and an alias resolving to the
#                                        primary would double-count one quota
MEASURED_RPM: dict[str, int] = {
    # 15 rpm tier
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,
    "gemini-3.1-flash-lite-preview": 15,
    # 10, not 15 -- the exception that made a per-model table necessary
    "gemini-2.5-flash-lite": 10,
    # 5 rpm tier
    "gemini-3.6-flash": 5,
    "gemini-3.5-flash": 5,
    "gemini-3-flash-preview": 5,
    # Gemma never tripped at 26 requests, so 30 is a floor rather than a
    # measurement. Left at the documented value.
    "gemma-4-26b-a4b-it": 30,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- environment ---
    # Drives anything that must behave differently in production. Typed as a
    # Literal so a typo ("prd") fails at startup instead of silently falling
    # through to whatever the permissive branch happens to be.
    #
    # Defaults to "prod", which looks backwards for a dev-first app and is the
    # entire point: a new deployment that forgets to set APP_ENV gets the
    # RESTRICTIVE behaviour. Fail closed. Local development sets it explicitly
    # in docker-compose.yml, where forgetting it is instantly obvious.
    app_env: Literal["dev", "prod"] = "prod"

    # --- model profile ---
    # Which bundle from MODEL_PROFILES supplies the model defaults. Switchable
    # PER REQUEST in dev only -- see `use_model_profile` and the guard in the
    # turn endpoints. In prod this is whatever the environment says and nothing
    # can move it, which is why the override lives in the API layer rather than
    # in this class.
    model_profile: Literal["gemini", "gemma"] = "gemini"

    @property
    def supports_tool_calling(self) -> bool:
        """Does the active profile's model emit native functionCall parts?

        The ReAct loop is built on them. Gemma has none -- asked to use tools it
        narrates its intentions as prose, which `generate_tools` correctly reads
        as "no calls requested", so the loop exits on round 1 having retrieved
        nothing and `draft` answers from an empty evidence set.

        That failure is silent and looks like a bad answer rather than a
        misconfiguration, so the graph consults this instead of trusting the
        `react` flag. See `gather_strategy`.
        """
        return self.model_profile in _TOOL_CALLING_PROFILES

    def profile_conflicts(self) -> list[tuple[str, object, object]]:
        """Fields where an explicit setting contradicts the active profile.

        A profile's value is that its fields move TOGETHER -- the models, the
        token ceiling, the history strategy and the hop budgets are one coherent
        set. A single stale environment variable overriding one of them produces
        a mixture that is worse than either profile: measured here, a leftover
        `LLM_MODEL=...gemma...` in .env gave a Gemma workhorse with Gemini's
        250K-token budgets and Gemini's hop counts.

        Explicit settings still WIN -- silently ignoring what someone deliberately
        configured is the worse failure. But the contradiction is reported, so an
        incoherent mix is a line in the log rather than a mystery.
        """
        profile = MODEL_PROFILES.get(self.model_profile, {})
        return [
            (field, getattr(self, field), expected)
            for field, expected in profile.items()
            if field in self.model_fields_set and getattr(self, field) != expected
        ]

    @model_validator(mode="after")
    def _apply_model_profile(self) -> "Settings":
        """Fill in the profile's values for anything not set explicitly.

        `model_fields_set` is the load-bearing part: it holds the fields that
        actually arrived from the environment or the constructor, so an explicit
        LLM_MODEL in .env still beats the profile. Without that check, switching
        profile would silently discard a developer's deliberate override -- and
        the app would keep reporting the model they asked for while calling a
        different one.
        """
        for field, value in MODEL_PROFILES.get(self.model_profile, {}).items():
            if field not in self.model_fields_set:
                setattr(self, field, value)
        return self

    @property
    def docs_enabled(self) -> bool:
        """OpenAPI docs are a dev convenience and a production information leak.

        Serving them publicly hands an attacker the full endpoint list, request
        schemas and the existence of admin-ish routes like /auth/claim. There is
        no staging environment yet; when one appears, this becomes a three-way
        decision (open / password-gated / absent) rather than a boolean.
        """
        return self.app_env == "dev"

    # --- Google AI Studio ---
    #
    # THREE models, and the split is measured rather than aesthetic. Free-tier
    # request limits are per model, so the question is not "which model is
    # best" but "which model can this call afford".
    #
    #   model                   rpm  schema      availability   used for
    #   gemini-3.6-flash          5  strict      3/3            the ANSWER
    #   gemini-3.5-flash-lite    15  strict      3/3            plan/clarify/critique
    #   gemma-4-26b              30  strict      ok             query rewriting only
    #   gemini-3.8-flash          5  strict      503 under load  --
    #   gemini-3.7-flash          5  --          503             --
    #   gemini-flash-latest       5  --          503 (0/3)       --
    #   gemini-3.5-flash          5  VIOLATED    ok              --
    #
    # Two findings worth keeping, both measured live on this key rather than
    # read off a docs page:
    #
    # 1. gemini-3.5-flash VIOLATES a responseSchema -- it returned "Here is the
    #    JSON requested:" as prose -- so it is unusable for structured output
    #    whatever its quota.
    # 2. The newest is not the most available. 3.8-flash passed an isolated
    #    schema test and then returned 503 for a real draft call; flash-latest
    #    was 503 on all three attempts. 3.6-flash was 3/3 on both counts, which
    #    is why it holds the one job a user actually reads.
    google_api_key: str = ""

    # The workhorse: plan, clarify, critique, summarise. 15 rpm is what makes a
    # 4-call agent turn possible -- at the full Flash models' 5 rpm, one turn
    # would consume an entire minute of budget and a critique retry would stall
    # in backoff.
    llm_model: str = "models/gemini-3.5-flash-lite"
    llm_tokens_per_minute: int = 250_000
    llm_requests_per_minute: int = 15

    # The stronger model, reserved for the text the user actually reads. One
    # call per draft, two if the critic sends it back -- which fits inside 5
    # rpm where a four-call turn would not.
    answer_model: str = "models/gemini-3.6-flash"
    answer_tokens_per_minute: int = 250_000
    answer_requests_per_minute: int = 5

    # ---- Parley: the audio-only app -------------------------------------
    #
    # STT is an ORDINARY generateContent call with an audio part, not a
    # dedicated speech endpoint -- which is why it names a normal model. The
    # lite model was measured transcribing its own synthesised speech back
    # correctly, and it is the cheapest thing on the list that does.
    #
    # There is also `models/gemini-3.5-transcribe-live`, which is bidi-only.
    # Streaming is the right shape for continuous dictation and the wrong shape
    # for push-to-talk turns, where the audio is complete before anything is
    # sent.
    voice_stt_model: str = "models/gemini-3.5-flash-lite"
    # TTS models are separate and DO need naming: `responseModalities: [AUDIO]`
    # is rejected by every ordinary model.
    #
    # A LIST, tried in order, for the same reason the answer pool is a list:
    # the newest is not the most available. Free-tier TTS quota is per model
    # and small, and it is genuinely normal for one of these to return 429
    # while another answers in three seconds -- measured, in that exact
    # configuration, mid-build.
    #
    # Falling back matters more here than anywhere else in the app: a turn
    # whose synthesis fails has nothing to show at all. Everywhere else a
    # degraded answer is still an answer; in an audio-only app it is silence.
    voice_tts_models: list[str] = [
        "models/gemini-3.1-flash-tts-preview",
        "models/gemini-2.5-flash-preview-tts",
        "models/gemini-2.5-pro-preview-tts",
    ]
    voice_default: str = "Kore"

    # THE NATIVE AUDIO MODEL. Parley's engine.
    #
    # Not a TTS model and not an STT model -- audio in and audio out, with no
    # transcript in the middle. Measured on this deployment, same question and
    # same corpus: the cascade took 12-53 seconds to the first sound, this
    # takes 2.0.
    #
    # `gemini-3.8-live` over the 2.5 native-audio models because it is the only
    # one that also accepts TEXT output, which makes it debuggable: a session
    # that misbehaves can be re-run in text mode to see what it would have
    # said. The 2.5 models reject TEXT outright.
    #
    # Live models are bidi-only (WebSocket) and, per the quota console, carry
    # unlimited requests per minute -- unlike the TTS models, which are the
    # constrained ones and which this replaces.
    live_model: str = "models/gemini-3.8-live"
    # Spoken answers are capped hard. A page of prose is a fine thing to read
    # and four minutes of unskippable audio to listen to -- there is no
    # scanning ahead in speech, so length costs the listener far more than it
    # costs a reader. The agent is asked for brevity in the prompt; this is the
    # backstop for when it does not comply.
    voice_answer_max_chars: int = 1_200

    # Query rewriting stays on Gemma, deliberately. It is the one call where
    # throughput beats quality: multi-query fires N rewrites per turn, the
    # output is short phrases rather than prose, and Gemma's 30 rpm is the
    # highest budget available. Nothing a user reads comes from this model.
    rewriter_model: str = "models/gemma-4-26b-a4b-it"
    rewriter_tokens_per_minute: int = 16_000
    rewriter_requests_per_minute: int = 30

    # Now accurate rather than aspirational: every Gemini Flash model on this
    # key reports a 1,048,576-token input window and supports native
    # functionDeclarations, verified live. Kept as response_schema because the
    # graph's nodes are not tool calls -- switching is now a real option rather
    # than something the model cannot do.
    llm_structured_mode: Literal["response_schema", "tool_calling"] = "response_schema"

    # --- judge model (evaluation Tier 2) ---
    # A DIFFERENT model from the one being evaluated, deliberately: a model
    # judging its own output has a documented self-preference bias, and Gemma
    # is also the weaker judge (no function calling, degenerates at moderate
    # temperature, and it is the component under test).
    #
    # The quota shapes are opposite, which is the other reason. Verified on
    # this key: gemini-3.5-flash-lite allows 250K tokens/minute against Gemma's
    # 16K, so judging -- which sends the answer plus every retrieved chunk --
    # fits comfortably where Gemma would spend a minute of budget per call.
    # Gemma's 14,400 requests/day dwarfs Flash Lite's 500, so small frequent
    # calls stay on Gemma and large ones move here.
    judge_model: str = "models/gemini-3.5-flash-lite"
    judge_requests_per_minute: int = 15
    judge_tokens_per_minute: int = 250_000

    # --- model pools ---
    # Extra models a role may ALSO use, comma-separated. Free-tier quota is per
    # model, so two models are two budgets and a role that can use either has
    # the sum -- this is addition for throughput, not failover for reliability.
    #
    # The primary is always tried first and only spills over when its own
    # budget is momentarily spent, so the secondary answers rarely and quality
    # stays consistent on the common path.
    #
    # Verified on this key (see `python -m app.scripts.probe_limits`):
    #   gemini-2.5-flash-lite     15 rpm, schema ok, one transient 503
    #   gemini-flash-lite-latest  15 rpm, schema ok
    #
    # `-latest` is deliberately NOT the shipped default. It is an alias, and if
    # it resolves to the primary the pool would hold two entries against ONE
    # real quota -- appearing to double the budget while doubling nothing.
    #
    # NEVER mix Gemma and Gemini in one pool. `ModelPool` refuses it at
    # construction: Gemma emits no functionCall parts, so a ReAct round landing
    # on a Gemma member retrieves nothing and reports success.
    answer_model_pool: str = "models/gemini-2.5-flash-lite"
    llm_model_pool: str = ""

    @property
    def answer_pool(self) -> list[str]:
        return [m.strip() for m in self.answer_model_pool.split(",") if m.strip()]

    @property
    def llm_pool(self) -> list[str]:
        return [m.strip() for m in self.llm_model_pool.split(",") if m.strip()]

    def limits_for(self, model: str) -> tuple[int, int]:
        # Measured per-model quota takes precedence over any role default.
        #
        # Pool members were previously given their ROLE's budget on the
        # assumption that same-family members share a tier. The probe disproved
        # it: gemini-2.5-flash-lite allows 10 requests/minute where every other
        # flash-lite allows 15, so inheriting the role's 15 handed it half again
        # its real quota and it would have collected the 429s the pool exists to
        # avoid.
        measured = MEASURED_RPM.get(model.removeprefix("models/"))
        if measured is not None:
            # Tokens still come from the role: the probe measures REQUEST rate,
            # and every Gemini model here reports the same 250K/minute.
            tokens = (
                self.rewriter_tokens_per_minute
                if model == self.rewriter_model
                else self.answer_tokens_per_minute
            )
            return measured, tokens
        return self._role_limits(model)

    def _role_limits(self, model: str) -> tuple[int, int]:
        """(requests_per_minute, tokens_per_minute) for a model name.

        Each model gets its OWN limiter keyed on these numbers, because the
        free-tier request quota is per model -- 5/min for the full Flash
        models, 15 for Flash Lite, 30 for Gemma. Sharing one budget across them
        would throttle every call on the strictest limit and waste most of the
        combined quota.
        """
        name = model.removeprefix("models/")
        table = {
            self.answer_model: (
                self.answer_requests_per_minute,
                self.answer_tokens_per_minute,
            ),
            # POOL MEMBERS INHERIT THEIR ROLE'S BUDGET.
            #
            # Without this a member falls through to the `llm_*` default, which
            # is the WORKHORSE's budget and has nothing to do with it. Measured:
            # gemini-2.5-flash-lite (really 15 rpm) was being limited at 30,
            # because that is what the workhorse allowed -- so the pool member
            # would have been handed twice its real quota and collected the 429s
            # the pool exists to avoid.
            #
            # Inheriting the role's numbers is right because membership already
            # asserts interchangeability: a pool is same-family, same-tier by
            # construction. If a member ever genuinely differs, give it its own
            # entry rather than widening this.
            **{
                m: (self.answer_requests_per_minute, self.answer_tokens_per_minute)
                for m in self.answer_pool
            },
            **{
                m: (self.llm_requests_per_minute, self.llm_tokens_per_minute)
                for m in self.llm_pool
            },
            self.rewriter_model: (
                self.rewriter_requests_per_minute,
                self.rewriter_tokens_per_minute,
            ),
            self.judge_model: (
                self.judge_requests_per_minute,
                self.judge_tokens_per_minute,
            ),
        }
        for configured, limits in table.items():
            if name == configured.removeprefix("models/"):
                return limits
        return self.llm_requests_per_minute, self.llm_tokens_per_minute

    # --- embeddings ---
    embedding_provider: Literal["gemini", "fastembed"] = "gemini"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    # batchEmbedContents rejects >100 requests per call (measured: 250 -> 400).
    embedding_batch_size: int = 100
    embedding_requests_per_minute: int = 100
    embedding_tokens_per_minute: int = 30_000

    # --- retrieval ---
    # top_k of 5 is now a QUALITY choice, not a budget one.
    #
    # It was a budget one: Gemma allowed 16K tokens/minute, so a fat context
    # spent the whole minute on a single call. Gemini Flash Lite allows 250K
    # against a 1M window, so that constraint is gone and this could be raised.
    #
    # Left at 5 deliberately. Every recall/precision number in the evaluation
    # harness is measured at k in (1, 3, 5, 10), and moving the default silently
    # invalidates the comparison. Raise it when a measurement asks for it --
    # recall@10 is 0.951 against recall@5, so the headroom is real, but the
    # right fix for that gap is a reranker rather than a wider context.
    retrieval_top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150
    # Minimum BODY length (text minus the prefixed heading) for a chunk to be
    # kept. Measured problem: a heading with no body of its own produced a
    # 39-char chunk of pure title that ranked SECOND in every retrieval,
    # because a bare title embeds close to almost any question about the
    # document -- burning one of five slots on text no answer could cite.
    # 50 clears those while leaving genuinely short sections intact.
    min_chunk_chars: int = 50

    # Multi-query: rewrite the question into N variations, retrieve for each,
    # fuse the ranked lists with RRF. Costs one extra Gemma call plus N
    # embedding calls. Default off so /ask stays a true single-pass baseline;
    # the request can opt in per call.
    multi_query: bool = False
    query_variations: int = 3
    # Classify request intent (specific vs broad) even when multi_query is OFF.
    #
    # Scope and the query rewrites come from the same model call, so this costs
    # one Gemma call per turn. Worth it: without it, turning off multi-query
    # also turned off understanding the question, and "summarize this document"
    # fell back to a literal search for the word "summarize". The toggle now
    # controls fan-out, not comprehension.
    intent_always: bool = True
    # RRF's damping constant. 60 is the value from the original paper and the
    # default in Elasticsearch; larger flattens the weight given to rank 1.
    rrf_k: int = 60

    # --- hybrid retrieval ---
    # Fuse BM25 with the dense search. Each fails COMPLETELY in the other's
    # territory -- dense cannot find "INC-2024-1183", BM25 cannot match
    # "couldn't locate it" to "hard to find" -- so this is coverage, not a
    # tie-break.
    #
    # ON, and measured on the golden set (17 answerable questions, floor on):
    #
    #     recall@k     dense   +hybrid
    #     @1           0.520   0.490
    #     @3           0.725   0.814
    #     @5           0.873   0.931
    #     @10          0.971   1.000
    #
    # Note the SHAPE, not just the direction: recall rises at every k except 1,
    # where it falls. That is RRF behaving as designed -- it rewards agreement
    # across retrievers, so a chunk only one method ranked first gets pushed
    # down by two that agree on second. Hybrid buys breadth and costs a little
    # precision at the very top, which is exactly the job the reranker below
    # then does.
    hybrid_search: bool = True
    # Candidates each retriever contributes to the fusion. Deeper than the
    # final k on purpose: RRF only sees what each list contains, so a chunk at
    # rank 21 of a top-20 list contributes exactly nothing.
    hybrid_candidates: int = 20

    # --- absolute relevance floor ---
    # Minimum cosine SIMILARITY for a chunk to be considered at all. Qdrant
    # returns similarity (1.0 identical, 0.0 unrelated), which is the
    # complement of the distance the literature usually quotes.
    #
    # WHY THIS IS NEEDED AT ALL: RRF discards magnitudes and reads only rank,
    # so it always produces a confident top-k -- every candidate could be a
    # terrible match and the top 5 would look identical to a perfect run. The
    # floor is what makes "nothing relevant exists" a possible OUTCOME, which
    # is what the abstention path depends on.
    #
    # 0.0 disables it. 0.60 is CALIBRATED, not guessed -- the two score
    # distributions over the golden set, dense top-20:
    #
    #                 n    min    p05    median   max
    #     relevant    27   0.619  0.633  0.722    0.821
    #     irrelevant  313  0.514  0.541  0.616    0.781
    #
    #     floor   keeps relevant   drops irrelevant
    #     0.55    100.0%            7.7%
    #     0.60    100.0%           40.3%   <-- here
    #     0.62     96.3%           51.4%
    #     0.65     81.5%           67.1%
    #
    # 0.60 removes two fifths of the noise at zero recall cost; 0.62 starts
    # discarding real answers to gain another tenth, which is the wrong trade --
    # nothing downstream recovers from evidence that never arrived. Confirmed
    # against Tier 1: recall@1/3/5/10 identical with the floor on.
    #
    # DO NOT copy this number to another corpus. It is a property of this
    # embedding model at this dimension -- gemini-embedding-001 at 768 dims puts
    # everything in a narrow high band, so a threshold quoted from a paper using
    # a different model would either keep everything or discard everything.
    # Re-run the calibration instead.
    retrieval_score_floor: float = 0.60

    # --- reranking ---
    # An LLM reranker, listwise, with relevance GRADING merged into the same
    # call.
    #
    # Retrieval is a bi-encoder: query and chunk are embedded separately and
    # never meet, so the score is similarity between two summaries of meaning.
    # A reranker sees them TOGETHER and judges whether the passage answers the
    # question -- much better, far too slow to run over a whole corpus. Hence
    # retrieve broadly and cheaply, then rerank precisely.
    #
    # Listwise (one call ranking all candidates) rather than pointwise (one
    # call each): ranking is inherently comparative, and pointwise scoring
    # gives eight chunks the same 7/10.
    #
    # ON. It is the single largest measured win in the retrieval stack, and it
    # repairs the one thing hybrid made worse. Full ablation, golden set:
    #
    #                  dense   +floor   +hybrid   +rerank
    #     recall@1     0.520   0.520    0.490     0.578
    #     recall@3     0.725   0.725    0.814     0.951
    #     recall@5     0.892   0.873    0.931     1.000
    #     MRR@5        0.762   0.762    0.760     0.853
    #     MAP@5        0.698   0.690    0.691     0.845
    #     NDCG@5       0.763   0.752    0.769     0.890
    #     precision@5  0.235   0.224    0.235     0.282
    #
    # TREAT THESE AS APPROXIMATE. Both the query rewriter and the reranker are
    # model calls, and flash-lite ignores temperature=0 (the provider warns as
    # much), so consecutive runs of the same configuration move by a few points.
    # The ordering is stable across runs; a two-point difference is not a
    # result.
    #
    # The grading half earns its place separately: on the golden set's
    # UNANSWERABLE questions it returned an empty list ("CEO's total
    # compensation", "revenue in fiscal 2026"), which is what makes abstention
    # reachable at all. A reranker that always returns k cannot express "none of
    # these help".
    #
    # COST: one extra model call per retrieval, and the planned path retrieves
    # once per sub-question. That is affordable on the workhorse's budget and is
    # the main reason the gemma profile exists.
    rerank: bool = True
    # Candidates fed to the reranker. The model reorders only what it is given,
    # so this is the ceiling on recall; but models rank 100 items WORSE than 30
    # as position bias intensifies, so deeper is not strictly better.
    rerank_candidates: int = 20
    # Passage characters shown to the reranker. Enough to judge relevance,
    # short enough that 20 candidates do not become a 9K-token prompt.
    rerank_excerpt_chars: int = 800

    # --- parent-child retrieval ---
    # Search matches small chunks; the model reads the whole SECTION containing
    # them. Chunk size is otherwise one knob serving two opposed jobs --
    # retrieval wants small and sharp, generation wants surrounding context.
    #
    # A parent is derived, not stored: `chunking.py` already makes headings hard
    # boundaries, so (document_id, heading) identifies a section and no
    # migration or re-ingest is needed.
    #
    # OFF, and now off for a MEASURED reason rather than an unmeasured one.
    #
    # Full golden set, both arms on the gemma profile, zero judge errors:
    #
    #                        off     on
    #     faithfulness       0.882   0.873
    #     answer_relevancy   0.704   0.742
    #     context_precision  0.701   0.725
    #     context_recall     0.850   0.825
    #
    # Two up, two down, every delta <= 0.04 -- and consecutive runs of the SAME
    # configuration already move by a few points, because the rewriter and the
    # reranker are model calls and flash-lite ignores temperature=0. This is
    # noise, not a result. A four-question pilot had shown +0.215 on answer
    # relevancy; it collapsed to +0.038 over twenty, which is the ordinary fate
    # of a promising small sample.
    #
    # WHY IT DOES NOTHING HERE, AND WHEN IT WOULD. The corpus is 39 chunks and
    # most sections are one or two of them, so the chunk usually IS the section
    # and there is nothing to assemble: the retrieved-context counts barely
    # moved. Parent-child pays off on documents whose sections are many chunks
    # long -- build it for the tail, not the median. Re-measure when the corpus
    # grows; the mechanism is tested and ready.
    parent_retrieval: bool = False
    # Ceiling on one assembled section. A cap is needed because sections vary
    # wildly and five expanded parents could otherwise dwarf the token budget --
    # on the gemma profile, a minute's entire allowance.
    parent_max_chars: int = 6000

    # --- conversation history ---
    # Send the WHOLE transcript rather than a rolling summary plus the last
    # three exchanges.
    #
    # The old design existed because Gemma allowed 16K tokens per MINUTE across
    # 3-5 calls per turn, leaving ~1.5K for history -- about 8 plain turns
    # before compression became mandatory. Gemini Flash reports a 1,048,576
    # token input window and 250K tokens/minute, so that constraint is gone.
    #
    # This matters for correctness, not just convenience: a summary is a lossy
    # rewrite, and pronoun resolution ("and the prior year?") is exactly the
    # thing that breaks when the referent was compressed away.
    history_full: bool = True
    # Budget, not a limit -- deliberately far below the 1M window so history
    # can never crowd out retrieved passages, which are what the answer must
    # actually cite. Falls back to summary + recent turns beyond this.
    history_max_tokens: int = 200_000
    # Verbatim exchanges kept when history DOES have to be compressed.
    #
    # Was 6 (three exchanges), sized for Gemma leaving ~1.5K tokens for
    # history. On Gemini that floor is gone, so the fallback keeps twenty
    # exchanges rather than three -- compression should lose the distant past,
    # not last week.
    verbatim_messages: int = 40

    # --- web search (Serper) ---
    # Empty key = web search OFF, and the agent behaves exactly as it did
    # before: documents only, and an honest refusal when they do not cover the
    # question. Same self-configuring pattern as SUPABASE_URL and Langfuse.
    #
    # This does NOT relax grounding. A web result is a SOURCE that must be
    # cited, not licence to answer from memory -- the citation contract is
    # unchanged, some sources are just URLs rather than chunks.
    # ---- file storage -------------------------------------------------
    #
    # `local` by default, and that is a deliberate default rather than a
    # placeholder: object storage should not stand between cloning this repo
    # and seeing it work, and a backend that only runs with a paid account
    # configured is a backend nobody tests. See `services/storage.py`.
    storage_backend: str = "local"
    storage_dir: str = "/data/files"
    # ANY S3-compatible endpoint. Supabase is the recommendation -- 1GB, no
    # credit card, and already a dependency here for auth:
    #   https://<project>.supabase.co/storage/v1/s3
    # Cloudflare R2 is https://<account>.r2.cloudflarestorage.com and has far
    # more room, but requires a card to enable.
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_bucket: str = ""
    # R2 ignores this; Supabase wants its project region.
    s3_region: str = "auto"
    # Optional. A bucket served from a custom domain can be linked directly;
    # without one, the API streams the bytes itself.
    s3_public_base: str = ""

    # ---- background jobs -------------------------------------------------
    # The queue is a Postgres table and the worker runs in this process. Off
    # switches the worker only -- jobs still queue, and are picked up whenever
    # a worker next runs, which is what makes moving it to its own process a
    # deployment change rather than a code one.
    jobs_worker_enabled: bool = True

    # ---- emotion analysis -------------------------------------------------
    # Dimensional (arousal/dominance/valence), not categorical. See
    # `services/emotion.py` for why that distinction is the whole design.
    #
    # OFF by default: it needs torch and transformers, which is about a
    # gigabyte of dependencies, and the model does not fit in Render's 512MB
    # free tier. Queued jobs simply wait until something can run them.
    emotion_analysis: bool = False
    emotion_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    serper_api_key: str = ""
    serper_requests_per_minute: int = 60
    web_search_results: int = 5

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.serper_api_key)

    # --- Groq: open-weights models on someone else's hardware ---
    #
    # NOT self-hosting, and the distinction is worth keeping straight in the
    # code as well as the docs. GroqCloud runs the weights on their LPUs and
    # exposes an HTTP API; nothing about the runtime, the batching, the KV
    # cache or the hardware is ours. What IS ours is the choice of an
    # open-weights model, which means the same weights could later be run on
    # our own GPU with vLLM and this client would keep working -- the API is
    # OpenAI-compatible at both ends.
    #
    # That compatibility is the reason this is a separate tiny client rather
    # than something bolted onto GenAIClient: Google's generateContent and
    # OpenAI's chat/completions differ in message shape, tool format and usage
    # accounting, and pretending they are one thing is how both end up wrong.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 120.0

    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)

    # --- NVIDIA: hosted NeMo speech models, free developer key ---
    #
    # A SECOND provider rather than a swap. Groq serves Whisper and nothing
    # else; NVIDIA serves the NeMo family -- Parakeet, Canary, and
    # `nemotron-asr-streaming`, which is genuinely streaming rather than the
    # batch endpoint Whisper sits behind. Their ASR NIM profiles also carry
    # Sortformer speaker diarization, which is the thing Whisper structurally
    # cannot do.
    #
    # NOT the same host as the LLM catalogue. `integrate.api.nvidia.com` serves
    # chat models and returns 404 for /audio/transcriptions -- measured, and
    # the reason an earlier reading of this concluded NVIDIA had no ASR at all.
    # Speech runs on NVCF, where `api.nvcf.nvidia.com` answers 401 rather than
    # 404: the route exists and wants a key.
    nvidia_api_key: str = ""
    nvidia_nvcf_url: str = "https://api.nvcf.nvidia.com/v2/nvcf"
    # Which hosted function to invoke. Left EMPTY on purpose: function ids are
    # per-model UUIDs that change as NVIDIA publishes and retires previews, so
    # a hardcoded default would be a 404 waiting to happen. `/playground/nvidia
    # /functions` lists what this key can actually reach.
    nvidia_asr_function_id: str = ""

    @property
    def nvidia_enabled(self) -> bool:
        return bool(self.nvidia_api_key)

    # --- ReAct research mode ---
    # A genuine tool-calling loop: the model chooses which tool to call, sees
    # the result, and decides what to call next. That is what makes multi-hop
    # work -- "find the competitors" then "look up each one" -- because the
    # second query cannot be written until the first returns.
    #
    # Distinct from the plan/retrieve/draft/critique graph, which plans every
    # lookup UP FRONT. Both are kept: the planned path is what every
    # recall/faithfulness number in the evaluation harness measures, and
    # replacing it would silently invalidate all of them.
    #
    # Hard cap on tool-calling rounds. Each round is one model call plus its
    # tools, so this is the difference between a multi-hop answer and an
    # unbounded loop spending quota.
    # ON by default, which is a product decision rather than a measured one.
    #
    # The planned path can only search the documents, so with it as the default
    # the corpus IS the boundary: any question the files do not cover comes
    # back as "not in the provided documents", even when the answer is one web
    # search away. ReAct is the only path that can route a question to where
    # its answer actually lives, so the assistant has to default to it to
    # behave as advertised.
    #
    # What this costs: one model call per round on top of the tools.
    #
    # What it must NOT cost is the evaluation baseline. The planned graph is
    # still there and still the thing "agent" means in the harness, so every
    # caller that measures or compares it now passes `react=False` EXPLICITLY
    # -- tier2.py, /research and probe_hitl.py. Flipping this default without
    # those would have quietly changed what the recorded numbers refer to.
    react_default: bool = True
    react_max_rounds: int = 6
    # Cap on tools executed per round, so one greedy response cannot fan out
    # into dozens of searches.
    react_max_calls_per_round: int = 4

    # --- agent (step 4) ---
    # Hard cap on critique -> retrieve cycles. Each iteration costs ~2 Gemma
    # calls; unbounded self-critique is the easiest way to burn a daily quota
    # by accident, and in practice a third pass rarely finds anything a second
    # one missed.
    agent_max_iterations: int = 2
    agent_max_subquestions: int = 3
    # Classify each turn's INTENT before treating it as a search.
    #
    # Without it every message is a retrieval question, so 'remember to always
    # search the web too' gets answered by searching the documents FOR that
    # preference -- measured, and the answer was 'your documents do not mention
    # personal preferences regarding search behaviour'.
    #
    # Costs one workhorse call per turn. Cheap against the 55 rpm pool, and it
    # fails soft to 'ask', so a broken router never stops a question being
    # answered.
    agent_route: bool = True
    # Draft rewrites after an `unsupported_claim` verdict. A SEPARATE budget
    # from `agent_max_iterations`: a rewrite costs one call and no retrieval,
    # so charging it to the retrieval cycle would let one over-claim consume the
    # turn's ability to search.
    #
    # One, not two. A second pass over the same evidence rarely differs, and a
    # critic that rejects the rewrite twice is usually disagreeing about tone
    # rather than about support.
    agent_max_regens: int = 1

    # Output ceiling for the drafted answer.
    #
    # Config rather than a literal in `draft`, because it is the setting most
    # sensitive to which model is answering. On Gemini 2000 is comfortable; on
    # Gemma it is a third of an entire minute's token budget, and Gemma falls
    # into repetition loops that run until the cap -- so a generous ceiling
    # there buys nothing but a longer loop to salvage.
    #
    # It was hardcoded at 2000, and 900 before that. 900 silently truncated
    # answers past their `sources_used` list, which is emitted last, producing
    # "no sources cited" on answers that cited in every sentence.
    draft_max_output_tokens: int = 2000

    # --- human-in-the-loop ---
    # Ask the user a clarifying question when their request is too vague to
    # retrieve well, offering concrete options drawn from what their documents
    # actually contain.
    #
    # This is the one interrupt worth having in a RAG system. The agent has no
    # side effects to gate, so there is no "approve this action" moment -- but
    # there is a very common failure where the question genuinely does not say
    # enough to search on ("tell me about the pyramids"), and the model's only
    # alternative is to guess. Guessing wastes the whole turn; asking costs one
    # sentence.
    #
    # Default ON. It costs one extra Gemma call per turn to decide whether to
    # ask, which is cheap next to the 3-5 calls a misunderstood question wastes
    # -- and the node stays silent unless the request is genuinely too vague to
    # search, so most turns never see the pause.
    #
    # Programmatic callers are unaffected: a pause needs a thread to resume, and
    # `initial_state` forces this off when there is no thread_id. The evaluation
    # harness therefore never pauses and never pays for the check.
    agent_clarify: bool = True

    # --- observability (Langfuse) ---
    # Empty keys = tracing OFF, and the app behaves exactly as it did before
    # observability existed. Same pattern as SUPABASE_URL: a feature that
    # configures itself on rather than needing a separate flag.
    #
    # Self-hosted default. Point at https://cloud.langfuse.com for the hosted
    # service; nothing else changes.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://langfuse:3000"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # Fraction of turns to score with the LLM judge in the background. Judging
    # is expensive (~113s/question measured) so it CANNOT run inline -- and it
    # does not need to, because online scoring is for spotting drift across
    # many turns, not for grading each one.
    trace_score_sample_rate: float = 0.0

    # --- auth (Supabase as identity provider only) ---
    # Empty = auth disabled, and every request runs as an anonymous local user
    # with owner_id None. That keeps the app usable before keys are configured
    # and makes turning auth on a one-line change rather than a migration.
    supabase_url: str = ""
    # Only needed for legacy projects still on a shared HS256 secret. Projects
    # created after 2025-05-01 use asymmetric keys and need nothing here.
    supabase_jwt_secret: str = ""
    supabase_jwt_audience: str = "authenticated"

    # One-time migration switch for POST /auth/claim, which assigns rows with
    # owner_id IS NULL to the caller. It existed to adopt the corpus created
    # before auth, and that has been done.
    #
    # Default false on purpose: left enabled, any NEW account could claim any
    # ownerless rows that appeared later (for example if auth were briefly
    # disabled during maintenance). Enable it deliberately, run it once, turn
    # it off again.
    allow_claim_unowned: bool = False

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url)

    @property
    def supabase_jwks_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    # --- infra ---
    database_url: str = "postgresql+asyncpg://rd:rd_local_dev@localhost:5432/research_desk"

    # Connection budget. This app opens TWO independent pools against the same
    # database -- asyncpg via SQLAlchemy, and psycopg3 via LangGraph's
    # AsyncPostgresSaver -- so it uses roughly twice what a single-driver app
    # of the same size would.
    #
    # The trap: SQLAlchemy's pool_size is NOT a ceiling. max_overflow defaults
    # to 10, so the previous `pool_size=10` could actually open 20 connections,
    # and every value here is per *process*. Render running two instances
    # doubles it again. Hence explicit and small: worst case is
    # 5 + 2 + 2 = 9 connections per instance.
    db_pool_size: int = 5
    db_max_overflow: int = 2
    checkpointer_pool_size: int = 2
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def fastembed_model(self) -> str:
        return "BAAI/bge-small-en-v1.5"


# The active profile override, per request/task rather than per process.
#
# A ContextVar and not a module global: the API serves concurrent requests on
# one event loop, so a global would let one developer's "run this on Gemma"
# change the model for everybody else's in-flight turn. ContextVars are copied
# into each task, so the override reaches every `get_settings()` call inside
# that turn -- nodes, retrieval, the limiter -- and nothing outside it.
_profile_override: ContextVar[str | None] = ContextVar(
    "model_profile_override", default=None
)


@lru_cache
def _base_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=len(MODEL_PROFILES) + 1)
def _profile_settings(profile: str) -> Settings:
    """Settings re-read under a different profile. Cached: constructing these
    parses .env every time, and `get_settings()` is called several times per
    node."""
    return Settings(model_profile=profile)


def get_settings() -> Settings:
    override = _profile_override.get()
    if override is not None and override != _base_settings().model_profile:
        return _profile_settings(override)
    return _base_settings()


@contextmanager
def use_model_profile(profile: str | None) -> Iterator[None]:
    """Run a block under a different model profile.

    DEV ONLY, and the caller enforces that -- this function deliberately has no
    opinion about `app_env`, so the check sits at the API boundary where the
    request is, rather than being buried here where it would be easy to assume
    and hard to see.

    A token is reset in `finally` rather than setting the var back to None: the
    latter would clobber an enclosing override instead of restoring it.
    """
    if profile is None or profile not in MODEL_PROFILES:
        yield
        return

    token = _profile_override.set(profile)
    try:
        yield
    finally:
        _profile_override.reset(token)
