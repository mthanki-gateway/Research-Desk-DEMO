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
    google_api_key: str = ""
    llm_model: str = "models/gemma-4-26b-a4b-it"
    llm_tokens_per_minute: int = 16_000
    llm_requests_per_minute: int = 30
    # Gemma 4 ignores tool/functionDeclarations (it narrates tool use as prose)
    # but honours generationConfig.responseSchema. Every structured agent step
    # goes through a schema; bare JSON mode leaks chain-of-thought.
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

        Each model gets its OWN limiter keyed on these numbers. Sharing one
        budget across models with opposite shapes would throttle both on the
        wrong limits and waste most of the combined quota.
        """
        if model.removeprefix("models/") == self.judge_model.removeprefix("models/"):
            return self.judge_requests_per_minute, self.judge_tokens_per_minute
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
    # top_k stays small on purpose: Gemma 4's 16K tokens/minute means a fat
    # context window would spend the whole minute's budget on one call.
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
    # RRF's damping constant. 60 is the value from the original paper and the
    # default in Elasticsearch; larger flattens the weight given to rank 1.
    rrf_k: int = 60

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
