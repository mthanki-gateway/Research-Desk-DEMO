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
    ("GET", "/live/conversations"),
    ("GET", "/live/conversations/{conversation_id}"),
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
