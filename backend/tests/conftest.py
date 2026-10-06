"""Shared fixtures."""

import pytest

from app.services import keys


@pytest.fixture
def with_keys():
    """The caller has their own key for every provider.

    API keys are REQUIRED now (REQUIRE_USER_KEYS), so a feature is on only
    when the person has added its key -- a server key no longer counts. Tests
    about what a feature DOES, rather than whether it is unlocked, run as a
    person who has every key.
    """
    token = keys._active.set({p: f"test-{p}-key" for p in keys.PROVIDERS})
    yield
    keys._active.reset(token)
