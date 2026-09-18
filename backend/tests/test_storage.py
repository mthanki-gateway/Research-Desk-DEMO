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
    """`t/<owner>/docs/<uuid><ext>` -- by tenant, then by id, never by name."""

    def test_it_keys_by_tenant_then_id(self):
        did = uuid.UUID("11111111-2222-3333-4444-555555555555")
        assert storage.key_for("user-7", did, "Quarterly Report.PDF") == (
            f"t/user-7/docs/{did}.pdf"
        )

    def test_two_owners_never_share_a_prefix(self):
        """The property every other use of the prefix rests on.

        Deleting a tenant, measuring one, exporting one and writing a bucket
        policy for one are all a prefix match -- and all of them are wrong the
        moment two owners can land under the same path.
        """
        did = uuid.uuid4()
        a = storage.key_for("owner-a", did, "x.pdf")
        b = storage.key_for("owner-b", did, "x.pdf")
        assert a != b
        assert not a.startswith(b.rsplit("/docs", 1)[0] + "/")

    def test_anonymous_gets_a_named_segment(self):
        """Not "None", and not an empty segment.

        `owner_id` is None with auth off. An empty segment collapses the path
        and puts anonymous uploads at the root of the tenant space, where a
        prefixed delete for any tenant could reach them.
        """
        did = uuid.uuid4()
        for owner in (None, "", "   "):
            key = storage.key_for(owner, did, "x.pdf")
            assert key == f"t/{storage.ANONYMOUS_TENANT}/docs/{did}.pdf"
            assert "//" not in key
            assert "None" not in key

    def test_an_owner_id_cannot_reparent_the_path(self):
        """Ids are UUIDs, so this should never fire -- which is why it exists.

        A key is built once and stored for ever. An id containing a slash
        would silently move every file that user uploads into somebody else's
        prefix, and nothing downstream would notice.
        """
        did = uuid.uuid4()
        key = storage.key_for("../../owner-b", did, "x.pdf")
        assert key.startswith("t/")
        assert ".." not in key
        assert key.count("/docs/") == 1

    def test_the_filename_never_reaches_the_key(self):
        """Two files called report.pdf are two documents, not one.

        Keying by name collides on the second upload, and a name in a path is
        how traversal gets in.
        """
        did = uuid.uuid4()
        key = storage.key_for("user-7", did, "../../etc/passwd")
        assert ".." not in key
        assert "passwd" not in key

    def test_a_missing_extension_is_fine(self):
        did = uuid.uuid4()
        assert storage.key_for("user-7", did, "README") == f"t/user-7/docs/{did}"

    def test_recordings_share_the_tenant_prefix(self):
        """So one delete removes a tenant's documents and their audio together."""
        sid = uuid.uuid4()
        key = storage.media_key_for("user-7", sid, 3, "wav")
        assert key == f"t/user-7/howl/{sid}/0003.wav"
        assert key.startswith("t/user-7/")

    def test_recording_turns_sort_lexicographically(self):
        """A listing is ordered by key, and turn 10 must not precede turn 2."""
        sid = uuid.uuid4()
        keys = [storage.media_key_for("u", sid, n, "wav") for n in (2, 10, 1)]
        assert sorted(keys) == [keys[2], keys[0], keys[1]]


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
