"""Every route the frontend calls is actually registered.

WHY THIS EXISTS

A refactor of `_locale` rewrote `playground.py` by truncating the file at the
old function and appending the new one -- silently deleting the two NVIDIA
endpoints that lived below it. Ruff passed, 411 tests passed, and the only
symptom was the UI reporting "NVIDIA_API_KEY is not set" for a key that was
set, because a 404 and a disabled provider look identical from the browser.

Nothing here tests behaviour. It tests that the door exists, which is the
class of mistake that costs an hour and cannot be caught by reading the
diff of the file that broke.
"""

import asyncio

import pytest

from app import main
from app.main import app

# Paths the frontend calls. Kept as a literal list rather than derived from
# the router, which would make the test agree with whatever the code does.
REQUIRED = {
    ("GET", "/health"),
    ("GET", "/playground/status"),
    ("GET", "/playground/models"),
    ("POST", "/playground/complete"),
    ("POST", "/playground/transcribe"),
    ("GET", "/playground/nvidia/status"),
    ("GET", "/playground/nvidia/functions"),
    ("GET", "/profile/memory"),
    ("GET", "/sessions"),
    ("POST", "/sessions"),
    ("GET", "/documents"),
    ("GET", "/corpus/atlas"),
    ("GET", "/voice/status"),
    ("GET", "/live/status"),
    ("GET", "/live/profile-fields"),
    ("POST", "/live/howler"),
    ("POST", "/live/howler/preview"),
    ("GET", "/howler/projects"),
    ("POST", "/howler/projects"),
    ("GET", "/howler/projects/{project_id}/invites"),
    ("POST", "/howler/projects/{project_id}/invites"),
    ("DELETE", "/howler/invites/{invite_id}"),
    ("GET", "/live/conversations"),
    ("GET", "/live/conversations/{conversation_id}"),
    ("PATCH", "/live/conversations/{conversation_id}"),
    ("POST", "/voice/ask"),
    ("GET", "/corpus/atlas/ray/{message_id}"),
}


def _registered() -> set[tuple[str, str]]:
    """Served paths, read from the OpenAPI schema.

    NOT by walking `app.routes`: an included router appears there as a single
    object with no `.path` of its own, so a naive walk sees seven mystery
    entries and reports every real endpoint as missing -- which is exactly the
    false alarm this test would otherwise raise on a perfectly healthy app.
    The schema is also what any client actually reads.
    """
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


def test_every_route_the_frontend_calls_is_registered():
    missing = REQUIRED - _registered()
    assert not missing, f"routes the UI calls but the API does not serve: {sorted(missing)}"


def test_the_atlas_route_is_registered():
    """The Atlas page has one dependency and this is it."""
    assert ("GET", "/corpus/atlas") in _registered()


# ---------------------------------------------------------------------------
# Startup must not be able to prevent the port opening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBootIsBounded:
    """A dependency being down must not stop the service existing.

    Render kills a deploy that never opens a port, and reports it as "Port scan
    timeout reached, no open ports detected" -- which says nothing about the
    cause. Startup used to `await create_tables()` and `await
    init_checkpointer()` unguarded, so an unreachable Postgres meant uvicorn
    never bound and that opaque line was the only evidence.

    Binding and reporting a specific failure beats never binding: the log names
    what is broken, and /health repeats it on every request.
    """

    async def test_a_hanging_step_gives_up(self, monkeypatch):
        monkeypatch.setattr(main, "BOOT_TIMEOUT_SECONDS", 1)
        main.BOOT_FAILURES.clear()

        async def never_returns():
            await asyncio.sleep(3600)

        assert await main._boot_step("database", never_returns()) is None
        assert "timed out" in main.BOOT_FAILURES["database"]

    async def test_a_failing_step_is_recorded_not_raised(self):
        main.BOOT_FAILURES.clear()

        async def explodes():
            raise RuntimeError("connection refused")

        assert await main._boot_step("database", explodes()) is None
        assert "connection refused" in main.BOOT_FAILURES["database"]

    async def test_a_working_step_returns_its_value(self):
        main.BOOT_FAILURES.clear()

        async def fine():
            return "saver"

        assert await main._boot_step("checkpointer", fine()) == "saver"
        assert main.BOOT_FAILURES == {}

    async def test_failures_are_named_not_counted(self):
        """"Degraded" sends whoever is looking to find out which half."""
        main.BOOT_FAILURES.clear()

        async def explodes():
            raise RuntimeError("nope")

        await main._boot_step("database", explodes())
        await main._boot_step("checkpointer", explodes())
        assert set(main.BOOT_FAILURES) == {"database", "checkpointer"}
        main.BOOT_FAILURES.clear()
