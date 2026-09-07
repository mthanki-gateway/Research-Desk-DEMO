"""Conversation history: whole transcript vs the compression fallback.

The compression machinery (rolling summary + last N verbatim) was mandatory
under a 16K tokens/minute budget. On Gemini Flash -- 1M window, 250K/minute --
the whole transcript fits, and a summary is a lossy rewrite of exactly the
thing pronoun resolution depends on. These tests pin both paths.
"""

import pytest

from app.config import Settings
from app.db.models import Role
from app.services import history


class FakeMessage:
    """Minimal stand-in: `_render` only reads `role` and `content`."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeSession:
    def __init__(self, summary=None, upto=0):
        self.summary = summary
        self.summarised_upto = upto


def _conversation(n_exchanges: int) -> list[FakeMessage]:
    out = []
    for i in range(n_exchanges):
        out.append(FakeMessage(Role.user, f"question number {i}"))
        out.append(FakeMessage(Role.assistant, f"answer number {i}"))
    return out


def _settings(**overrides) -> Settings:
    return Settings(**{"google_api_key": "x", **overrides})


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setattr(history, "get_settings", lambda: _settings(history_full=True))


@pytest.fixture
def compressed(monkeypatch):
    # The verbatim window is pinned EXPLICITLY rather than left at the default.
    # It was raised from 6 to 40 when the model changed, which silently made
    # this test's 20-exchange conversation fit entirely inside the window --
    # so the assertion that old turns get dropped stopped testing anything.
    monkeypatch.setattr(
        history,
        "get_settings",
        lambda: _settings(history_full=False, verbatim_messages=6),
    )


class TestFullHistory:
    def test_empty_conversation(self, full):
        assert history.build_chat_context(FakeSession(), []) == ""

    def test_sends_every_message(self, full):
        """The whole point: nothing is dropped while it fits."""
        messages = _conversation(30)
        out = history.build_chat_context(FakeSession(), messages)
        assert "question number 0" in out, "the oldest turn was dropped"
        assert "question number 29" in out
        assert out.count("question number") == 30

    def test_ignores_a_stale_summary(self, full):
        """A summary is a lossy rewrite. With the transcript present it is
        redundant, and including both would double-count the early turns."""
        out = history.build_chat_context(
            FakeSession(summary="OLD SUMMARY"), _conversation(3)
        )
        assert "OLD SUMMARY" not in out

    def test_falls_back_when_over_budget(self, monkeypatch):
        """The budget is a real ceiling, so the fallback must still work."""
        monkeypatch.setattr(
            history,
            "get_settings",
            lambda: _settings(
                history_full=True, history_max_tokens=50, verbatim_messages=6
            ),
        )
        messages = _conversation(200)
        out = history.build_chat_context(FakeSession(summary="THE SUMMARY"), messages)
        assert "THE SUMMARY" in out
        assert "question number 0" not in out, "fallback still sent everything"
        # Only the verbatim tail survives.
        assert "question number 199" in out


class TestCompressionFallback:
    def test_summary_plus_recent_only(self, compressed):
        out = history.build_chat_context(
            FakeSession(summary="THE SUMMARY"), _conversation(20)
        )
        assert "THE SUMMARY" in out
        assert "question number 0" not in out
        assert "question number 19" in out


@pytest.mark.asyncio
class TestSummariserSkipped:
    async def test_no_model_call_when_history_is_whole(self, full):
        """Summarising costs a model call per eviction. With the transcript
        being sent whole there is nothing to compress, so that call must not
        happen -- it would be paid on every ordinary conversation for nothing."""
        session = FakeSession(summary="EXISTING", upto=4)
        summary, upto = await history.update_summary(session, _conversation(40))
        assert (summary, upto) == ("EXISTING", 4)
