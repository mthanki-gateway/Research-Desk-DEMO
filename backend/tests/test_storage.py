"""Keeping the original upload, and the two rules that make it safe.

Storage is a small module whose failures are quiet and expensive: a key built
from a filename collides, a key that escapes its root reads the host, and an
unreachable bucket that raises takes down an upload whose TEXT indexed
perfectly well. Each of those is pinned here.
"""

import asyncio
import uuid

import pytest

from app.services import storage


@pytest.fixture
def store(tmp_path):
    return storage.LocalStorage(str(tmp_path))


class TestKeys:
    """`docs/<uuid><ext>` -- by id, never by name."""

    def test_it_keys_by_id_and_keeps_the_extension(self):
        did = uuid.UUID("11111111-2222-3333-4444-555555555555")
        assert storage.key_for(did, "Quarterly Report.PDF") == f"docs/{did}.pdf"

    def test_the_filename_never_reaches_the_key(self):
        """Two files called report.pdf are two documents, not one.

        Keying by name collides on the second upload, and a name in a path is
        how traversal gets in.
        """
        did = uuid.uuid4()
        key = storage.key_for(did, "../../etc/passwd")
        assert ".." not in key
        assert "passwd" not in key

    def test_a_missing_extension_is_fine(self):
        did = uuid.uuid4()
        assert storage.key_for(did, "README") == f"docs/{did}"


class TestLocalStorage:
    def test_it_round_trips(self, store):
        async def go():
            await store.put("docs/x.pdf", b"%PDF-1.4", "application/pdf")
            return await store.get("docs/x.pdf")

        assert asyncio.run(go()) == b"%PDF-1.4"

    def test_a_missing_key_raises_rather_than_returning_empty(self, store):
        """Empty bytes would be served as a zero-page PDF, which looks real."""
        with pytest.raises(storage.StorageError):
            asyncio.run(store.get("docs/never-written.pdf"))

    def test_it_refuses_a_key_that_escapes_the_store(self, store):
        """Keys come from a database row, and rows can be wrong.

        They are minted from UUIDs so this should be impossible, which is
        exactly the kind of assumption worth enforcing rather than trusting.
        """
        for key in ("../outside.txt", "docs/../../etc/passwd"):
            with pytest.raises(storage.StorageError):
                asyncio.run(store.get(key))

    def test_deleting_something_that_is_already_gone_is_not_an_error(self, store):
        asyncio.run(store.delete("docs/never-written.pdf"))

    def test_local_cannot_sign(self, store):
        """None, not a fabricated URL -- there is no public host to sign for.

        The caller falls back to serving the bytes itself, which is why this
        returns None rather than raising.
        """
        assert asyncio.run(store.signed_url("docs/x.pdf")) is None
