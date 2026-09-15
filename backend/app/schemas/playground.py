from pydantic import BaseModel, Field


class PlaygroundStatus(BaseModel):
    enabled: bool
    default_model: str
    # Shown in the UI because it is the thing that would change when moving
    # from GroqCloud to a self-hosted vLLM, and seeing it makes that swap
    # concrete rather than abstract.
    base_url: str


class ModelInfo(BaseModel):
    id: str
    owned_by: str | None = None
    context_window: int | None = None


class ModelsOut(BaseModel):
    models: list[ModelInfo] = []


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    # Empty means "use the configured default", so the UI does not have to know
    # the default in order to send.
    model: str = ""
    system: str = ""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=32_000)


class CompletionOut(BaseModel):
    text: str
    finish_reason: str | None = None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: int = 0
    tokens_per_second: float | None = None


class TranscriptionOut(BaseModel):
    text: str
    # What Whisper detected, which is not necessarily what was requested.
    # Surfaced so a transcript in the wrong script has a visible cause.
    language: str | None = None
    duration_seconds: float = 0.0
    elapsed_ms: int = 0
    # Audio seconds transcribed per wall-clock second. Above 1 is the
    # condition for keeping up with a live speaker at all.
    realtime_factor: float | None = None
    # Whisper's own confidence. Shown, not trusted: on pure silence it
    # returns "Thank you." with no_speech_prob 0.000. See groq_client.
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    model: str
