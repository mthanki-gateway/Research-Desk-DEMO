"""Re-uploading the same bytes must not re-ingest them.

WHY THIS EXISTS

Nothing prevented duplicates before, and a duplicate is not a tidiness
problem. It spends real embedding quota, and it puts two copies of every
passage in the vector store under DIFFERENT chunk ids -- so the retrieval-level
dedupe in `state.py` cannot collapse them and a top-k slot is silently spent on
a passage the model has already been handed.

The two things pinned here are both cases where the feature would appear to
work and do nothing at all. Neither is visible in a diff, and neither produces
an error: the upload succeeds, the library looks right, and the duplicate is
created anyway.
"""

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.api.documents import _owner_match
from app.db.models import Document
from app.db.session import _MIGRATIONS


def _compiled(owner_id: str | None) -> str:
    """The WHERE clause `_find_by_digest` actually sends, as literal SQL.

    Built from the shipped predicate rather than a copy of it -- a test that
    reimplements the logic agrees with itself and not with the code.
    """
    stmt = select(Document).where(
        Document.content_sha256 == "abc", _owner_match(owner_id)
    )
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class TestOwnerScoping:
    def test_an_anonymous_owner_is_matched_with_is_null(self):
        """`owner_id = NULL` is NULL, never true.

        Auth is off today, so EVERY row has a NULL owner. Written with `==`
        the lookup matches nothing, every upload looks new, and the whole
        feature is dead in precisely the configuration it ships in -- while
        passing any test that remembers to set an owner.
        """
        sql = _compiled(None)
        assert "owner_id IS NULL" in sql
        assert "owner_id = NULL" not in sql

    def test_a_real_owner_is_matched_by_equality(self):
        assert "owner_id = 'u-1'" in _compiled("u-1")

    def test_the_digest_alone_is_never_the_whole_predicate(self):
        """Scoped by owner even though a SHA-256 is globally unique.

        Returning another tenant's row would both leak that they hold the file
        and hand this caller a document they cannot delete.
        """
        assert "owner_id" in _compiled(None)


class TestUniqueIndex:
    """The index is the backstop for two requests racing past the SELECT."""

    def test_the_index_coalesces_a_null_owner(self):
        """Without COALESCE the index would never once fire.

        SQL treats NULLs as DISTINCT from one another, so a plain
        UNIQUE (content_sha256, owner_id) imposes no constraint at all while
        every row has a NULL owner -- which is today.
        """
        ddl = " ".join(_MIGRATIONS)
        assert "documents_content_owner_uniq" in ddl
        assert "COALESCE(owner_id, '')" in ddl

    def test_the_index_ignores_rows_with_no_digest(self):
        """Rows ingested before the column existed cannot be backfilled.

        Their bytes were never stored. Without the partial clause they would
        all share a NULL digest and -- once COALESCE removes the NULL escape
        hatch -- collide with each other, so creating the index would fail on
        any database that already holds documents.
        """
        ddl = " ".join(_MIGRATIONS)
        assert "WHERE content_sha256 IS NOT NULL" in ddl

    def test_the_migrations_are_idempotent(self):
        """They run on EVERY boot, not once.

        There is no Alembic here; `create_all` cannot alter an existing table,
        so this is the migration. A statement that is not IF NOT EXISTS turns
        the second startup into a crash loop.
        """
        for statement in _MIGRATIONS:
            assert "IF NOT EXISTS" in statement, statement


def test_the_column_is_wide_enough_for_a_sha256():
    """64 hex characters. A VARCHAR(32) would truncate and collide."""
    assert Document.__table__.c.content_sha256.type.length == 64
