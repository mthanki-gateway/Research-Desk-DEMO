"""Facts ABOUT the data, as opposed to facts found IN it.

WHY THIS IS NOT RETRIEVAL

Every existing tool answers "what does the material say?" by embedding a query
and ranking passages. None of them can answer "how many documents are there",
"what is the average conversation length", or "what do these files broadly
cover" -- those are questions about the shape of the corpus, and semantic
search is the wrong instrument for them. Asked "what documents do you have?",
the agent could only deflect, while the UI listed all five by name beside it.

These are plain aggregate SQL reads: no embedding, no LLM call, and cheap
enough to call freely.

WHY THE RESULTS ARE NOT CITABLE

A retrieved passage is evidence, and the drafter must cite it [n] so a reader
can check the claim against its source. A count computed by this module is not
evidence of anything -- it IS the fact, produced by the application from its
own database, and there is no passage to point at. So these results travel
separately from `evidence` and the drafter is told to state them plainly
without a citation marker. Forcing a [n] onto them would mean either a
hallucinated citation or a silently dropped fact.

EVERY QUERY IS OWNER-SCOPED

Aggregates are the easiest place to leak across tenants: a count that quietly
includes another user's documents is wrong in a way nobody notices, because
the number still looks plausible. `owner_id` is therefore a required argument
on every function here, never a default.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import func, select

from app.db.models import ChatSession, Chunk, Document, Message
from app.db.session import SessionLocal

log = structlog.get_logger()

# Enough to describe a corpus without turning one tool result into the whole
# prompt. A user with 400 documents does not need all 400 filenames to learn
# what they have.
MAX_DOCUMENTS = 60
MAX_HEADINGS = 80


async def documents(owner_id: str | None) -> list[dict[str, Any]]:
    """One row per document: what it is called and how big it is.

    Includes documents that are still indexing or have FAILED, with their
    status. Hiding them would make "why can't you find anything in my report?"
    unanswerable -- the answer is usually that the report never finished
    indexing, and that is exactly the fact worth surfacing.
    """
    async with SessionLocal() as db:
        # Chars come from the chunks rather than `size_bytes`: the file size
        # includes markup and encoding overhead, so it is a poor proxy for how
        # much text there actually is to search.
        chars = (
            select(
                Chunk.document_id.label("document_id"),
                func.coalesce(func.sum(Chunk.n_chars), 0).label("n_chars"),
            )
            .group_by(Chunk.document_id)
            .subquery()
        )
        stmt = (
            select(Document, chars.c.n_chars)
            .outerjoin(chars, chars.c.document_id == Document.id)
            .order_by(Document.created_at.desc())
            .limit(MAX_DOCUMENTS)
        )
        if owner_id is not None:
            stmt = stmt.where(Document.owner_id == owner_id)
        rows = (await db.execute(stmt)).all()

    return [
        {
            # Needed to SCOPE a search to this document. Without it the agent
            # can learn a filename and then has no way to act on it.
            "id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status.value if hasattr(doc.status, "value") else doc.status,
            "n_chunks": doc.n_chunks,
            "n_pages": doc.n_pages,
            "n_chars": int(n_chars or 0),
            "uploaded": doc.created_at.date().isoformat() if doc.created_at else None,
        }
        for doc, n_chars in rows
    ]


async def headings(owner_id: str | None) -> list[str]:
    """Section headings across the whole corpus, deduped, in document order.

    THE CHEAP ROUTE TO "WHAT ARE THESE DOCUMENTS ABOUT".

    The alternative is reading every document and summarising, which costs a
    model call per document and a large context. Headings are what the author
    already wrote to describe their own sections, they are stored by the
    chunker, and they are one indexed query -- so a question about themes can
    be answered from the structure the documents came with rather than by
    re-deriving it.
    """
    async with SessionLocal() as db:
        stmt = (
            select(Chunk.heading)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.heading.isnot(None))
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .limit(MAX_HEADINGS * 4)  # room to dedupe below
        )
        if owner_id is not None:
            stmt = stmt.where(Document.owner_id == owner_id)
        rows = (await db.execute(stmt)).scalars().all()

    seen: set[str] = set()
    out: list[str] = []
    for h in rows:
        key = (h or "").strip()
        if key and key.lower() not in seen:
            seen.add(key.lower())
            out.append(key)
        if len(out) >= MAX_HEADINGS:
            break
    return out


async def corpus_stats(owner_id: str | None) -> dict[str, Any]:
    """Aggregates over the documents: counts, size, spread, dates."""
    async with SessionLocal() as db:
        doc_stmt = select(
            func.count(Document.id),
            func.coalesce(func.sum(Document.n_chunks), 0),
            func.min(Document.created_at),
            func.max(Document.created_at),
        )
        chunk_stmt = select(
            func.coalesce(func.sum(Chunk.n_chars), 0),
            func.coalesce(func.avg(Chunk.n_chars), 0),
            func.count(Chunk.id),
        ).join(Document, Document.id == Chunk.document_id)
        status_stmt = select(Document.status, func.count(Document.id)).group_by(
            Document.status
        )

        if owner_id is not None:
            doc_stmt = doc_stmt.where(Document.owner_id == owner_id)
            chunk_stmt = chunk_stmt.where(Document.owner_id == owner_id)
            status_stmt = status_stmt.where(Document.owner_id == owner_id)

        n_docs, n_chunks, first, last = (await db.execute(doc_stmt)).one()
        total_chars, avg_chunk_chars, real_chunks = (await db.execute(chunk_stmt)).one()
        by_status = {
            (s.value if hasattr(s, "value") else str(s)): n
            for s, n in (await db.execute(status_stmt)).all()
        }

    return {
        "n_documents": int(n_docs or 0),
        "by_status": by_status,
        # `n_chunks` on Document is what the ingester recorded; counting the
        # chunk rows is what actually exists. They disagree when an ingest
        # failed part way, and that gap is worth being able to see.
        "n_chunks_recorded": int(n_chunks or 0),
        "n_chunks_present": int(real_chunks or 0),
        "total_characters": int(total_chars or 0),
        "avg_characters_per_chunk": round(float(avg_chunk_chars or 0), 1),
        "avg_characters_per_document": (
            round(int(total_chars or 0) / n_docs, 1) if n_docs else 0
        ),
        "first_upload": first.date().isoformat() if first else None,
        "latest_upload": last.date().isoformat() if last else None,
    }


async def conversation_stats(owner_id: str | None) -> dict[str, Any]:
    """Aggregates over the chat sessions: how many, how long, how recent."""
    async with SessionLocal() as db:
        session_stmt = select(
            func.count(ChatSession.id),
            func.min(ChatSession.created_at),
            func.max(ChatSession.updated_at),
        )
        # Averages are computed PER SESSION and then averaged, not taken over
        # all messages at once: "the average conversation" is a statement about
        # conversations, and a flat average over messages would let one very
        # long chat dominate and report a mean no actual session resembles.
        per_session = (
            select(func.count(Message.id).label("n"))
            .join(ChatSession, ChatSession.id == Message.session_id)
            .group_by(Message.session_id)
        )
        length_stmt = select(
            func.count(Message.id),
            func.coalesce(func.avg(func.length(Message.content)), 0),
        ).join(ChatSession, ChatSession.id == Message.session_id)

        if owner_id is not None:
            session_stmt = session_stmt.where(ChatSession.owner_id == owner_id)
            per_session = per_session.where(ChatSession.owner_id == owner_id)
            length_stmt = length_stmt.where(ChatSession.owner_id == owner_id)

        n_sessions, first, last = (await db.execute(session_stmt)).one()
        counts = (await db.execute(per_session)).scalars().all()
        n_messages, avg_chars = (await db.execute(length_stmt)).one()

        # Split by role in one pass rather than two round trips.
        role_stmt = (
            select(
                Message.role,
                func.count(Message.id),
                func.coalesce(func.avg(func.length(Message.content)), 0),
            )
            .join(ChatSession, ChatSession.id == Message.session_id)
            .group_by(Message.role)
        )
        if owner_id is not None:
            role_stmt = role_stmt.where(ChatSession.owner_id == owner_id)
        by_role = {
            (r.value if hasattr(r, "value") else str(r)): {
                "count": n,
                "avg_characters": round(float(avg or 0), 1),
            }
            for r, n, avg in (await db.execute(role_stmt)).all()
        }

    ordered = sorted(counts)
    return {
        "n_sessions": int(n_sessions or 0),
        # Sessions with no messages are counted above but contribute 0 here,
        # which is why this is not just n_messages / n_sessions: a pile of
        # empty sessions from clicking "New chat" would otherwise drag the
        # average down and make conversations look shorter than they are.
        "n_sessions_with_messages": len(ordered),
        "n_messages": int(n_messages or 0),
        "avg_messages_per_session": (
            round(sum(ordered) / len(ordered), 1) if ordered else 0
        ),
        "median_messages_per_session": (
            ordered[len(ordered) // 2] if ordered else 0
        ),
        "longest_session_messages": ordered[-1] if ordered else 0,
        "avg_message_characters": round(float(avg_chars or 0), 1),
        "by_role": by_role,
        "first_session": first.date().isoformat() if first else None,
        "last_activity": last.date().isoformat() if last else None,
    }


def _line(label: str, value: Any) -> str:
    return f"- {label}: {value}"


def render_documents(rows: list[dict[str, Any]], outline: list[str]) -> str:
    """The document list as the model should read it."""
    if not rows:
        return (
            "No documents have been uploaded yet, so there is nothing indexed "
            "to search."
        )

    lines = [f"{len(rows)} document(s):"]
    for r in rows:
        bits = [f"{r['n_chunks']} chunks", f"{r['n_chars']:,} characters"]
        if r["n_pages"]:
            bits.append(f"{r['n_pages']} pages")
        if r["status"] != "ready":
            # Loud, because a document that is not ready is invisible to every
            # search -- the single most useful thing this tool can report.
            bits.append(f"STATUS {r['status'].upper()} -- not searchable")
        if r["uploaded"]:
            bits.append(f"uploaded {r['uploaded']}")
        lines.append(f"- {r['filename']} ({', '.join(bits)})")

    if outline:
        lines.append("")
        lines.append(
            "Section headings across these documents, which is what they "
            "cover:"
        )
        lines.extend(f"- {h}" for h in outline)
    return "\n".join(lines)


def render_stats(title: str, stats: dict[str, Any]) -> str:
    """A stats dict as flat labelled lines, nested dicts inlined."""
    lines = [title]
    for key, value in stats.items():
        label = key.replace("_", " ")
        if isinstance(value, dict):
            if not value:
                continue
            inner = ", ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(_line(label, inner))
        else:
            lines.append(_line(label, value))
    return "\n".join(lines)


async def resolve_document(name: str, owner_id: str | None) -> tuple[str, str] | None:
    """Find a document id from a filename the model produced.

    FORGIVING ON PURPOSE. The agent gets filenames from `list_documents` and
    then retypes one, so it arrives with the extension dropped, the case
    changed, or only the distinctive part of the name. An exact match would
    fail on all three and the failure would look like "that document does not
    exist", which is a worse answer than searching everything.

    Returns (id, actual filename) so the caller can tell the model which
    document it really got -- a fuzzy match that silently resolves to the
    wrong file is the failure mode this ordering is designed to surface.
    """
    wanted = name.strip().lower()
    if not wanted:
        return None

    rows = await documents(owner_id)
    # Exact, then prefix, then substring. Ordered most specific first so a
    # request for "acme-report" cannot be captured by a longer name that
    # merely contains it.
    for row in rows:
        if row["filename"].lower() == wanted:
            return row["id"], row["filename"]
    for row in rows:
        if row["filename"].lower().startswith(wanted):
            return row["id"], row["filename"]
    for row in rows:
        if wanted in row["filename"].lower():
            return row["id"], row["filename"]
    return None
