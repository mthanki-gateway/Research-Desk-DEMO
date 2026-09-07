from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    def limits_for(self, model: str) -> tuple[int, int]:
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
    serper_api_key: str = ""
    serper_requests_per_minute: int = 60
    web_search_results: int = 5

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.serper_api_key)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
