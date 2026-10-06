"""Bring-your-own-keys: whose key a call uses, and what happens without one."""

import pytest

from app.services import keys


class TestWhoseKey:
    def test_the_persons_key_wins(self):
        token = keys._active.set({"serper": "mine"})
        try:
            assert keys.key_for("serper") == "mine"
        finally:
            keys._active.reset(token)

    def test_no_server_fallback_when_keys_are_required(self, monkeypatch):
        """The whole point of requiring keys: usage is theirs, not ours."""
        real = keys.get_settings()
        monkeypatch.setattr(
            keys, "get_settings",
            lambda: real.model_copy(update={"require_user_keys": True, "serper_api_key": "server"}),
        )
        token = keys._active.set({})
        try:
            assert keys.key_for("serper") == ""
        finally:
            keys._active.reset(token)

    def test_server_key_is_the_fallback_when_not_required(self, monkeypatch):
        real = keys.get_settings()
        monkeypatch.setattr(
            keys, "get_settings",
            lambda: real.model_copy(update={"require_user_keys": False, "serper_api_key": "server"}),
        )
        token = keys._active.set({})
        try:
            assert keys.key_for("serper") == "server"
        finally:
            keys._active.reset(token)


class TestMissingKey:
    def test_require_raises_a_message_the_person_can_act_on(self):
        token = keys._active.set({})
        try:
            with pytest.raises(keys.KeyMissing) as err:
                keys.require("groq")
            assert "Settings" in str(err.value)
            assert err.value.provider == "groq"
        finally:
            keys._active.reset(token)

    @pytest.mark.asyncio
    async def test_the_shared_clients_send_the_persons_key(self):
        """Hooks set the header per request, from whoever the call is for."""
        import httpx

        hook = keys.header_hook("gemini", "x-goog-api-key")
        token = keys._active.set({"gemini": "AIza-mine"})
        try:
            req = httpx.Request("POST", "https://example.invalid")
            await hook(req)
            assert req.headers["x-goog-api-key"] == "AIza-mine"
        finally:
            keys._active.reset(token)
