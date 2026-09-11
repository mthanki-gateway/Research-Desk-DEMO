"""Parent-child retrieval: search small, read whole.

THE TENSION THIS RESOLVES

Chunk size is one knob serving two opposed jobs. Retrieval wants SMALL chunks --
a 900-character passage about one thing embeds to a sharp point, where a whole
section embeds to a blur of four topics and matches nothing well. Generation
wants LARGE ones -- an answer needs the surrounding sentences, the figure the
paragraph refers back to, the caveat two paragraphs down.

Parent-child stops it being one knob. Search matches children; the model reads
the parent that contains them.

WHAT A PARENT IS HERE

The SECTION. `chunking.py` already treats markdown headings as hard boundaries,
so a chunk can never straddle two sections -- which is exactly the invariant
this needs, and it is already enforced upstream rather than being something to
check. Measured on the current corpus, `## Timeline` in the incident
post-mortem is two chunks; assembling them gives the model the whole timeline
instead of half of it.

NO PARENT TABLE, AND NO PARENT COLUMN EITHER

A parent is a VIEW over its children: `SELECT ... WHERE document_id = ? AND
heading = ? ORDER BY chunk_index`. Storing it would mean a migration and a
second thing to keep in step with re-chunking; deriving it means re-chunking
cannot desynchronise the two, because there is only one.

THE DEDUPE IS THE POINT, NOT A TIDY-UP

Relevant passages cluster, so several winning children routinely share one
parent. Without dedupe the model receives the same section three times and
concludes the document "repeatedly emphasises" it -- a fabricated finding
produced entirely by a plumbing bug. The key is IDENTITY (document + heading),
never similarity: two different sections making the same point is a real
finding, and collapsing those would erase it.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import structlog
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chunk
from app.db.session import SessionLocal
from app.services.vectorstore import SearchHit

log = structlog.get_logger()


def parent_key(hit: SearchHit) -> str:
    """Which parent a child belongs to.

    Web results are their own parent: there is no document to expand within,
    and a url is already a whole page.

    A chunk with NO heading -- most PDF extractions -- is its own parent too.
    Grouping every headingless chunk of a document together would assemble the
    entire file, which is not a parent but a failure to have one.
    """
    if hit.source != "document":
        return f"web:{hit.url or hit.chunk_id}"
    if not hit.heading:
        return f"chunk:{hit.chunk_id}"
    return f"doc:{hit.document_id}:{hit.heading}"


def _strip_overlap(previous: str, current: str, max_overlap: int) -> str:
    """Remove the tail of `previous` repeated at the head of `current`.

    Adjacent chunks share `chunk_overlap` characters BY DESIGN -- it is what
    stops a sentence split across a boundary being lost to both sides. That
    overlap is doing its job in the embedding; concatenated back into a parent
    it is just a duplicated paragraph, and duplicated text in a prompt reads as
    emphasis.

    Longest-match-first so the true overlap wins rather than a short
    coincidental repeat.
    """
    limit = min(max_overlap, len(previous), len(current))
    for size in range(limit, 20, -1):
        if previous.endswith(current[:size]):
            return current[size:].lstrip()
    return current


async def _fetch_children(
    document_id: uuid.UUID, heading: str
) -> list[tuple[int, str]]:
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Chunk.chunk_index, Chunk.text)
                .where(Chunk.document_id == document_id, Chunk.heading == heading)
                .order_by(Chunk.chunk_index)
            )
        ).all()
    return [(i, t) for i, t in rows]


async def read_around(
    chunk_id: uuid.UUID,
    *,
    before: int = 1,
    after: int = 1,
    owner_id: str | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """Neighbouring passages around one already retrieved, in document order.

    SCOPE IS ENFORCED IN SQL, NOT TRUSTED FROM THE ARGUMENT. `chunk_id` arrives
    from the model, which can hallucinate one or echo an id it saw elsewhere. A
    lookup that resolved it before checking ownership would let a guessed id
    read another tenant's document -- so the owner filter is part of the query
    that finds the anchor, and an id outside the caller's scope simply does not
    resolve.
    """
    from app.db.models import Document

    before = max(0, min(3, before))
    after = max(0, min(3, after))

    async with SessionLocal() as db:
        anchor_stmt = (
            select(Chunk.document_id, Chunk.chunk_index)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id == chunk_id)
        )
        if owner_id is not None:
            anchor_stmt = anchor_stmt.where(Document.owner_id == owner_id)
        if document_ids:
            anchor_stmt = anchor_stmt.where(Chunk.document_id.in_(document_ids))

        anchor = (await db.execute(anchor_stmt)).first()
        if anchor is None:
            return []

        document_id, index = anchor
        rows = (
            await db.execute(
                select(
                    Chunk.id,
                    Chunk.chunk_index,
                    Chunk.heading,
                    Chunk.page,
                    Chunk.text,
                    Document.filename,
                )
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    Chunk.document_id == document_id,
                    Chunk.chunk_index >= index - before,
                    Chunk.chunk_index <= index + after,
                )
                .order_by(Chunk.chunk_index)
            )
        ).all()

    return [
        SearchHit(
            chunk_id=cid,
            document_id=document_id,
            filename=filename,
            page=page,
            chunk_index=idx,
            heading=heading,
            text=text,
            # Neighbours were not scored against the query -- they were asked
            # for by position. A similarity score here would be a fiction, and
            # 0.0 keeps them below the floor if they are ever re-filtered.
            score=0.0,
            meta={"read_around": True, "anchor": str(chunk_id)},
            found_by=["read_around"],
        )
        for cid, idx, heading, page, text, filename in rows
    ]


async def expand_to_parents(hits: list[SearchHit]) -> list[SearchHit]:
    """Replace each hit with the assembled section containing it.

    RANK ORDER IS PRESERVED and the best child speaks for its parent: the
    parent takes the position and score of the highest-ranked child that
    landed in it. Re-sorting by anything else would discard the reranker's
    judgement, which is the most informed ordering available.

    Degrades to the input on any failure. An expansion that throws would lose
    retrieval that already succeeded, and the un-expanded chunks are exactly
    what this function had before it existed.
    """
    settings = get_settings()
    if not settings.parent_retrieval or not hits:
        return hits

    out: list[SearchHit] = []
    seen: set[str] = set()
    budget = settings.parent_max_chars

    for hit in hits:
        key = parent_key(hit)
        if key in seen:
            # A second child of a parent already assembled. Dropped, not
            # appended -- see the module docstring on false corroboration.
            continue
        seen.add(key)

        if not key.startswith("doc:"):
            out.append(hit)
            continue

        try:
            children = await _fetch_children(hit.document_id, hit.heading or "")
        except Exception as exc:  # noqa: BLE001 - never lose good retrieval
            log.warning("parent_fetch_failed", error=str(exc), chunk=str(hit.chunk_id))
            out.append(hit)
            continue

        if len(children) <= 1:
            # The chunk IS the section. Nothing to assemble, and re-using the
            # hit keeps its text identical to what retrieval scored.
            out.append(hit)
            continue

        parts: list[str] = []
        total = 0
        truncated = False
        for _, text in children:
            piece = _strip_overlap(parts[-1], text, settings.chunk_overlap) if parts else text
            if total + len(piece) > budget:
                # Capped, and SAID SO. A silent drop here is the kind of bug
                # that looks like a model failure: the answer omits something
                # the section plainly contains, and nothing in the trace
                # explains why.
                truncated = True
                break
            parts.append(piece)
            total += len(piece)

        assembled = "\n".join(parts)
        if truncated:
            log.info(
                "parent_truncated",
                heading=hit.heading,
                kept=len(parts),
                of=len(children),
                chars=total,
            )
            assembled += "\n[section continues]"

        out.append(
            replace(
                hit,
                text=assembled,
                # meta records the expansion so a trace can show that the model
                # read a section rather than the chunk that was scored.
                meta={**hit.meta, "parent_children": len(parts), "parent": hit.heading},
            )
        )

    collapsed = len(hits) - len(out)
    if collapsed:
        log.info("parents_deduped", hits=len(hits), parents=len(out), collapsed=collapsed)
    return out
