"""Deleting an account: everything the person owns, everywhere.

SCOPED BY THE CALLER, NEVER BY A PARAMETER. `delete_account` takes the
owner id the auth layer derived from the verified token, and the endpoint
that calls it has no id in its path or body -- so there is nothing a request
could change to point it at somebody else's data.

REFUSED WITHOUT AN OWNER. With auth disabled every caller is ANONYMOUS and
owner_id is None, and in this codebase None means "unscoped": a delete
filtered on it would match every row that has no owner, which in a dev stack
is all of them. That is not an account; there is nothing to delete.

ORDER MATTERS. Files and vectors first, because their only index is the rows
that are about to go -- delete a document row before its stored file and
nothing can ever find the file again (the same leak `ingest.delete_document`
documents). Then the rows, children before parents where no cascade covers
them.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import delete, select

from app.db.models import (
    ChatSession,
    Document,
    GraphEntity,
    GraphHydration,
    GraphRelation,
    HowlerProject,
    Job,
    Message,
    Preference,
    TopicEdge,
    TopicNode,
    UserApiKey,
)
from app.db.session import SessionLocal
from app.services.storage import get_storage

log = structlog.get_logger()


class NoAccount(RuntimeError):
    """There is no owner to delete (auth disabled / anonymous)."""


async def delete_account(owner_id: str | None) -> dict[str, Any]:
    if not owner_id:
        raise NoAccount("There is no account to delete while signed out or with auth disabled.")

    from app.services import atlas
    from app.services.ingest import delete_document
    from app.services.lexical import invalidate_lexical_index

    counts: dict[str, int] = {}

    # 1. Documents, through the one path that already removes each one's
    #    vectors, stored original, chunks and graph rows together.
    async with SessionLocal() as db:
        doc_ids = list(
            (await db.execute(select(Document.id).where(Document.owner_id == owner_id))).scalars()
        )
    for d in doc_ids:
        await delete_document(d)
    counts["documents"] = len(doc_ids)

    # 2. Interview recordings. Their only index is the message rows.
    async with SessionLocal() as db:
        metas = list(
            (
                await db.execute(
                    select(Message.agent_meta)
                    .join(ChatSession, ChatSession.id == Message.session_id)
                    .where(ChatSession.owner_id == owner_id)
                )
            ).scalars()
        )
    storage = get_storage()
    recordings = 0
    for meta in metas:
        key = (meta or {}).get("audio_key")
        if key:
            try:
                await storage.delete(key)
                recordings += 1
            except Exception as exc:  # noqa: BLE001 - keep going; report below
                log.warning("account_recording_delete_failed", error=str(exc)[:120])
    counts["recordings"] = recordings

    # 3. Every owned row. Sessions cascade to their messages and their
    #    session-scoped preferences; projects cascade to their invites.
    async with SessionLocal() as db:
        for name, model in (
            ("conversations", ChatSession),
            ("preferences", Preference),
            ("howler_projects", HowlerProject),
            ("graph_entities", GraphEntity),
            ("graph_relations", GraphRelation),
            ("hydrations", GraphHydration),
            ("topics", TopicNode),
            ("topic_links", TopicEdge),
            ("jobs", Job),
            ("api_keys", UserApiKey),
        ):
            result = await db.execute(delete(model).where(model.owner_id == owner_id))
            counts[name] = result.rowcount or 0
        await db.commit()

    invalidate_lexical_index()
    atlas.invalidate()
    log.info("account_deleted", owner=owner_id[:8], **counts)
    return counts
