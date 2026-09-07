"""The tracing seam.

What matters here is not that Langfuse works -- that is their tests -- but that
THIS APP is unaffected when it doesn't. An observability layer that can break a
request is worse than no observability, because you added a system to find out
when things go wrong and it became the thing that goes wrong.

So every test is a variation on "tracing is broken; does the app still behave".
"""

import pytest

from app.config import Settings
from app.services import tracing


@pytest.fixture(autouse=True)
def _clean():
    """Drop the cached client between tests.

    `get_tracer` caches, including caching a failure, so without this the first
    test's state decides every later one.
    """
    tracing.reset_for_tests()
    yield
    tracing.reset_for_tests()


# Every Langfuse key must be passed EXPLICITLY.
#
# `Settings()` reads the process environment, so a bare
# `Settings(langfuse_public_key="pk")` inherits the real LANGFUSE_SECRET_KEY
# from the container and comes out ENABLED -- which made these tests pass or
# fail depending on whether the local observability stack happened to be
# running. Pinning both fields is what makes them deterministic.
def _settings(**overrides) -> Settings:
    # Defaults MERGED, not passed alongside, so an override of the same key is
    # a replacement rather than a duplicate keyword argument.
    return Settings(
        **{
            "google_api_key": "x",
            "langfuse_public_key": "",
            "langfuse_secret_key": "",
            **overrides,
        }
    )


@pytest.fixture
def off(monkeypatch):
    """Tracing unconfigured -- the default, and the state that must be safe."""
    monkeypatch.setattr(tracing, "get_settings", _settings)


@pytest.fixture
def broken(monkeypatch):
    """Configured, but the SDK raises on every call."""

    class Exploding:
        def __getattr__(self, _name):
            def boom(*args, **kwargs):
                raise RuntimeError("langfuse is down")

            return boom

    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(langfuse_public_key="pk", langfuse_secret_key="sk"),
    )
    monkeypatch.setattr(tracing, "_client", Exploding())
    monkeypatch.setattr(tracing, "_checked", True)


class TestDisabled:
    def test_not_enabled_without_keys(self, off):
        assert tracing.enabled() is False

    def test_tracer_is_none(self, off):
        assert tracing.get_tracer() is None

    def test_no_callbacks(self, off):
        """An empty list, not None -- it is spread into a LangGraph config."""
        assert tracing.callbacks() == []

    def test_turn_is_a_working_context_manager(self, off):
        """The app wraps real work in this, so it must always yield."""
        with tracing.turn(name="t", session_id="s", user_id="u"):
            pass

    def test_observe_yields_none(self, off):
        """Call sites must tolerate a None handle; that is the whole seam."""
        with tracing.observe("x") as span:
            assert span is None

    def test_update_tolerates_none(self, off):
        tracing.update(None, output="anything")

    def test_no_trace_id(self, off):
        assert tracing.current_trace_id() is None

    def test_score_is_silent(self, off):
        tracing.score("faithfulness", 0.9)

    def test_flush_is_silent(self, off):
        tracing.flush()


class TestBrokenBackend:
    """Every one of these would raise if the seam did not swallow errors."""

    def test_turn_still_yields(self, broken):
        entered = False
        with tracing.turn(name="t"):
            entered = True
        assert entered, "a tracing failure must not skip the work it wraps"

    def test_observe_still_yields(self, broken):
        with tracing.observe("x") as span:
            assert span is None

    def test_score_swallows(self, broken):
        tracing.score("x", 1.0, trace_id="t")

    def test_flush_swallows(self, broken):
        tracing.flush()

    def test_trace_id_returns_none(self, broken):
        assert tracing.current_trace_id() is None

    def test_update_swallows(self, broken):
        class Exploding:
            def update(self, **_):
                raise RuntimeError("nope")

        tracing.update(Exploding(), output="x")


class TestEnabling:
    def test_needs_both_keys(self):
        """One key alone is a misconfiguration, not a partial setup."""
        assert _settings().tracing_enabled is False
        assert _settings(langfuse_public_key="pk").tracing_enabled is False
        assert _settings(langfuse_secret_key="sk").tracing_enabled is False
        assert (
            _settings(
                langfuse_public_key="pk", langfuse_secret_key="sk"
            ).tracing_enabled
            is True
        )

    def test_init_is_attempted_once(self, monkeypatch):
        """A bad host must not cost a connection attempt per request.

        Counted through `get_settings`, which `get_tracer` calls exactly once
        per real initialisation -- so a second call reaching it would mean the
        result was not cached.
        """
        calls = {"n": 0}

        def counting_settings():
            calls["n"] += 1
            return _settings(langfuse_public_key="pk", langfuse_secret_key="sk")

        monkeypatch.setattr(tracing, "get_settings", counting_settings)

        tracing.get_tracer()
        after_first = calls["n"]
        tracing.get_tracer()
        tracing.get_tracer()
        assert calls["n"] == after_first, "re-initialised instead of caching"


class TestExceptionsPropagate:
    """A traced body that raises must surface ITS OWN exception.

    This is the regression guard for a real bug. Both helpers used to wrap the
    `yield` in `try/except Exception: yield`, so an exception raised by the
    caller -- which propagates back through the yield point -- was caught by
    the tracing layer, which then yielded a SECOND time. A @contextmanager
    generator must yield exactly once, so Python raised

        RuntimeError: generator didn't stop after throw()

    replacing the real error with a meaningless one. It reached the UI as a
    failed turn whose message said nothing about the actual failure.
    """

    def test_turn_propagates_when_disabled(self, off):
        with pytest.raises(ValueError, match="the real error"):  # noqa: PT012
            with tracing.turn(name="t"):
                raise ValueError("the real error")

    def test_turn_propagates_when_backend_is_broken(self, broken):
        with pytest.raises(ValueError, match="the real error"):  # noqa: PT012
            with tracing.turn(name="t"):
                raise ValueError("the real error")

    def test_observe_propagates_when_disabled(self, off):
        with pytest.raises(ValueError, match="the real error"):  # noqa: PT012
            with tracing.observe("x"):
                raise ValueError("the real error")

    def test_observe_propagates_when_backend_is_broken(self, broken):
        with pytest.raises(ValueError, match="the real error"):  # noqa: PT012
            with tracing.observe("x"):
                raise ValueError("the real error")

    def test_nested_propagates(self, broken):
        """The real shape: an observation inside a turn, innermost raising."""
        with pytest.raises(ValueError, match="the real error"):  # noqa: PT012
            with tracing.turn(name="t"), tracing.observe("inner"):
                raise ValueError("the real error")
