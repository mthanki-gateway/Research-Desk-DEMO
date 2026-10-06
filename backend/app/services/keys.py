"""API keys people bring, and which features each one unlocks.

ONE REGISTRY, read by both sides. The settings page renders its cards from
PROVIDERS -- label, what it powers, how to get one -- and the feature gates
are computed from the same entries, so a provider cannot be listed without
saying what depends on it.

THE SERVER'S KEYS ARE THE FALLBACK FOR NOW. A feature is available when the
person has added the key it needs OR the deployment has one. Setting
REQUIRE_USER_KEYS stops counting the server's, which is the switch for
clients paying for their own usage: every feature whose key they have not
added then reports itself unavailable and the UI greys it out.

STORED ENCRYPTED (Fernet, from KEY_ENCRYPTION_SECRET). The plaintext is only
ever decrypted server-side, by `resolve`, for the call that needs it; the API
returns the last four characters and nothing else.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import UserApiKey
from app.db.session import SessionLocal

PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {
        "label": "Google Gemini",
        "setting": "google_api_key",
        "features": ["chat", "library", "atlas", "parley", "lab"],
        "powers": "Chat answers, document embeddings, Atlas, the knowledge graph, and Parley's live voice.",
        "url": "https://aistudio.google.com/app/apikey",
        "steps": [
            "Open Google AI Studio and sign in with a Google account.",
            "Choose “Get API key”, then “Create API key”, and pick or create a Cloud project.",
            "Copy the key (it starts with “AIza”) and paste it here.",
        ],
        "prefix": "AIza",
    },
    "serper": {
        "label": "Serper (web search)",
        "setting": "serper_api_key",
        "features": ["web_search", "hydrate"],
        "powers": "Web search in chat, and “Hydrate from web knowledge” in the graph.",
        "url": "https://serper.dev/api-key",
        "steps": [
            "Sign up at serper.dev; new accounts include free searches.",
            "Open the API Key page from the dashboard.",
            "Copy the key and paste it here.",
        ],
        "prefix": "",
    },
    "groq": {
        "label": "Groq",
        "setting": "groq_api_key",
        "features": ["playground", "transcribe", "interview_transcription"],
        "powers": "The Playground, Transcribe, and the Whisper pass that transcribes Parley interviews.",
        "url": "https://console.groq.com/keys",
        "steps": [
            "Sign in at console.groq.com.",
            "Open API Keys and choose “Create API Key”.",
            "Copy it straight away (it is shown once, starts with “gsk_”) and paste it here.",
        ],
        "prefix": "gsk_",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "setting": "nvidia_api_key",
        "features": ["nvidia_speech"],
        "powers": "NVIDIA's hosted speech models (Parakeet, Canary) in Transcribe.",
        "url": "https://build.nvidia.com/",
        "steps": [
            "Sign in at build.nvidia.com.",
            "Open any model page and choose “Get API Key”, or go to your profile's API keys.",
            "Copy the key (it starts with “nvapi-”) and paste it here.",
        ],
        "prefix": "nvapi-",
    },
}

# Features whose need is "any of these" rather than one provider. Interview
# transcription runs on Groq Whisper when available and Gemini otherwise.
ANY_OF: dict[str, list[str]] = {"interview_transcription": ["groq", "gemini"]}


class KeysUnavailable(RuntimeError):
    """No encryption secret configured, so keys cannot be stored safely."""


def _fernet() -> Fernet:
    s = get_settings()
    secret = s.key_encryption_secret or s.supabase_jwt_secret
    if not secret:
        raise KeysUnavailable(
            "Saving keys needs KEY_ENCRYPTION_SECRET set on the server."
        )
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


def _owner(col, owner_id: str | None):
    return col == owner_id if owner_id is not None else col.is_(None)


def server_has(provider: str) -> bool:
    return bool(getattr(get_settings(), PROVIDERS[provider]["setting"], ""))


async def _rows(owner_id: str | None) -> dict[str, UserApiKey]:
    async with SessionLocal() as db:
        rows = (
            await db.execute(select(UserApiKey).where(_owner(UserApiKey.owner_id, owner_id)))
        ).scalars()
        return {r.provider: r for r in rows}


async def status(owner_id: str | None) -> dict[str, Any]:
    """Every provider's card, plus which features are available."""
    rows = await _rows(owner_id)
    require = get_settings().require_user_keys
    cards = []
    usable: set[str] = set()
    for pid, p in PROVIDERS.items():
        mine = rows.get(pid)
        server = server_has(pid)
        source = "yours" if mine else ("server" if server and not require else "none")
        if source != "none":
            usable.add(pid)
        cards.append(
            {
                "id": pid,
                "label": p["label"],
                "powers": p["powers"],
                "url": p["url"],
                "steps": p["steps"],
                "features": p["features"],
                "yours": {"last4": mine.last4, "updated_at": mine.updated_at.isoformat()}
                if mine
                else None,
                "server_fallback": server and not require,
                "source": source,
            }
        )
    features: dict[str, bool] = {}
    for pid, p in PROVIDERS.items():
        for f in p["features"]:
            features[f] = features.get(f, False) or pid in usable
    for f, any_of in ANY_OF.items():
        features[f] = any(p in usable for p in any_of)
    return {"providers": cards, "features": features, "require_user_keys": require}


async def save(owner_id: str | None, provider: str, key: str) -> None:
    if provider not in PROVIDERS:
        raise LookupError(provider)
    key = key.strip()
    token = _fernet().encrypt(key.encode()).decode()
    async with SessionLocal() as db:
        await db.execute(
            delete(UserApiKey).where(
                _owner(UserApiKey.owner_id, owner_id), UserApiKey.provider == provider
            )
        )
        db.add(UserApiKey(owner_id=owner_id, provider=provider, ciphertext=token, last4=key[-4:]))
        await db.commit()


async def remove(owner_id: str | None, provider: str) -> None:
    async with SessionLocal() as db:
        await db.execute(
            delete(UserApiKey).where(
                _owner(UserApiKey.owner_id, owner_id), UserApiKey.provider == provider
            )
        )
        await db.commit()


async def resolve(owner_id: str | None, provider: str) -> str | None:
    """The key to use for this owner's call: theirs, else the server's.

    Server-side only. Returns None when neither exists, or when
    REQUIRE_USER_KEYS is set and they have not added one.
    """
    row = (await _rows(owner_id)).get(provider)
    if row is not None:
        try:
            return _fernet().decrypt(row.ciphertext.encode()).decode()
        except (InvalidToken, KeysUnavailable):
            return None
    if get_settings().require_user_keys:
        return None
    return getattr(get_settings(), PROVIDERS[provider]["setting"], "") or None


# ---------------------------------------------------------------------------
# Runtime: which key THIS request, job or call should use.
#
# A context variable, set once at the edge -- the authenticated request, the
# job being run, the live socket -- and read by the provider clients just
# before they send. The repo's rule is that owner scoping for DATA is passed
# explicitly, never through a context var; this is not data scoping, it is
# credential selection deep inside shared HTTP clients that a dozen call
# chains reach, and threading an owner through every one of them is what
# would drift.
#
# Context vars are copied into tasks created from the request, so a
# background ingest started by an upload carries the uploader's keys too.
# ---------------------------------------------------------------------------

import contextvars  # noqa: E402

_active: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "provider_keys", default=None
)


class KeyMissing(RuntimeError):
    """A feature was used without the key it needs. Shown to the person."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        label = PROVIDERS.get(provider, {}).get("label", provider)
        super().__init__(f"Add your {label} API key in Settings to use this.")


async def bind(owner_id: str | None) -> None:
    """Load this owner's keys into the current context. Never raises."""
    found: dict[str, str] = {}
    try:
        f = _fernet()
        for pid, row in (await _rows(owner_id)).items():
            try:
                found[pid] = f.decrypt(row.ciphertext.encode()).decode()
            except InvalidToken:
                continue
    except Exception:  # noqa: BLE001 - no secret, or no table yet: no user keys
        found = {}
    _active.set(found)


def key_for(provider: str) -> str:
    """The key for this call: the person's, else (unless required) the server's."""
    mine = (_active.get() or {}).get(provider)
    if mine:
        return mine
    if get_settings().require_user_keys:
        return ""
    return getattr(get_settings(), PROVIDERS[provider]["setting"], "") or ""


def require(provider: str) -> str:
    """`key_for`, raising KeyMissing instead of returning ""."""
    key = key_for(provider)
    if not key:
        raise KeyMissing(provider)
    return key


def header_hook(provider: str, header: str, fmt: str = "{}"):
    """An httpx request hook that sets the provider key on every request.

    For the SHARED clients (Gemini LLM, embeddings, Serper), which are built
    once at startup and so cannot have a person's key baked into them.
    """

    async def hook(request) -> None:
        request.headers[header] = fmt.format(require(provider))

    return hook
