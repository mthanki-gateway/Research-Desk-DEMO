"""Topics: the person's own nodes in the knowledge graph, and exploring outward.

The document graph shows what the person's files say. Topics are for
LEARNING: add a concept, have it linked to what is already there, then
explore from any node to pull in the concepts around it -- and from those,
further. Each step is one model call; nothing is pre-generated.

Links made here are the model's general knowledge, not a document's claim,
and the UI says so: a topic edge has no passage to open. That is the trade
for being able to wander past what the documents cover.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import delete, or_, select

from app.db.models import GraphEntity, TopicEdge, TopicNode
from app.db.session import SessionLocal
from app.services.graph import CATEGORIES, TYPES, key_of
from app.services.pool import get_pool

log = structlog.get_logger()


class TopicError(RuntimeError):
    pass


def _owner(col, owner_id: str | None):
    return col == owner_id if owner_id is not None else col.is_(None)


LINK_SYSTEM = """<role>
The assistant places a new concept into a person's knowledge graph: it says \
what kind of thing it is and which existing nodes it genuinely connects to.
</role>

<task>
Given the new concept and the list of existing nodes, return its `type`, its \
`category`, and up to six links to existing nodes that a knowledgeable person \
would agree are real, direct connections, each with a short lowercase \
predicate read from the new concept to the node ("is a type of", "used in", \
"competes with", "invented by"). Use node names exactly as listed. No link is \
better than a weak one; an empty list is fine.
</task>"""

LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": TYPES},
        "category": {"type": "string", "enum": CATEGORIES},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "predicate": {"type": "string"},
                },
                "required": ["node", "predicate"],
            },
        },
    },
    "required": ["type", "category", "links"],
}

EXPLORE_SYSTEM = """<role>
The assistant helps a person learn by exploring outward from one concept in \
their knowledge graph.
</role>

<task>
Given a concept and what it already connects to, propose five to eight \
closely related concepts worth learning next: its parts, the ideas it builds \
on, what it is used for, what it is compared with, who or what made it, and \
what came before or after it. Prefer specific, named concepts over vague \
ones ("HBM2 memory" over "memory"). Skip anything already connected. Each \
gets a short lowercase predicate read FROM the explored concept TO the new \
one ("is built on", "succeeded by", "uses", "invented by"), a `type` and a \
`category`.
</task>"""

EXPLORE_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "predicate": {"type": "string"},
                    "type": {"type": "string", "enum": TYPES},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["name", "predicate", "type", "category"],
            },
        }
    },
    "required": ["concepts"],
}


async def _names(owner_id: str | None) -> dict[str, str]:
    """Every node this owner has, key -> display name."""
    async with SessionLocal() as db:
        ents = (
            await db.execute(
                select(GraphEntity.key, GraphEntity.name).where(_owner(GraphEntity.owner_id, owner_id))
            )
        ).all()
        tops = (
            await db.execute(
                select(TopicNode.key, TopicNode.name).where(_owner(TopicNode.owner_id, owner_id))
            )
        ).all()
    out: dict[str, str] = {}
    for k, n in [*ents, *tops]:
        if len(n) > len(out.get(k, "")):
            out[k] = n
    return out


async def _ask(prompt: str, system: str, schema: dict) -> dict:
    raw = await get_pool("llm").generate(
        prompt, system=system, schema=schema, temperature=0.3, max_output_tokens=1500
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TopicError("The model returned something unreadable; try again.") from exc


async def add_topic(owner_id: str | None, name: str) -> dict[str, Any]:
    name = " ".join(name.split())[:200]
    key = key_of(name)
    if not key:
        raise TopicError("Give the topic a name.")
    existing = await _names(owner_id)
    others = [n for k, n in existing.items() if k != key][:400]
    out = await _ask(
        f"New concept: {name}\n\nExisting nodes:\n" + "\n".join(f"- {n}" for n in others),
        LINK_SYSTEM,
        LINK_SCHEMA,
    )
    by_name = {key_of(n): k for k, n in existing.items()}
    async with SessionLocal() as db:
        found = (
            await db.execute(
                select(TopicNode).where(_owner(TopicNode.owner_id, owner_id), TopicNode.key == key)
            )
        ).scalar_one_or_none()
        if found is None:
            db.add(
                TopicNode(
                    owner_id=owner_id,
                    key=key,
                    name=name,
                    type=out.get("type") if out.get("type") in TYPES else "concept",
                    category=out.get("category") if out.get("category") in CATEGORIES else "Other",
                    origin="added",
                )
            )
        links = 0
        for link in out.get("links") or []:
            target = by_name.get(key_of(str(link.get("node") or "")))
            pred = str(link.get("predicate") or "").strip().lower()[:128]
            if target and target != key and pred:
                db.add(TopicEdge(owner_id=owner_id, source_key=key, target_key=target, predicate=pred))
                links += 1
        await db.commit()
    log.info("topic_added", key=key, links=links)
    return {"key": key, "links": links}


async def explore(owner_id: str | None, key: str) -> dict[str, Any]:
    existing = await _names(owner_id)
    name = existing.get(key)
    if not name:
        raise LookupError("No such node.")
    async with SessionLocal() as db:
        near = (
            await db.execute(
                select(TopicEdge.source_key, TopicEdge.target_key).where(
                    _owner(TopicEdge.owner_id, owner_id),
                    or_(TopicEdge.source_key == key, TopicEdge.target_key == key),
                )
            )
        ).all()
    neighbours = {existing.get(t if s == key else s, "") for s, t in near} - {""}
    out = await _ask(
        f"Concept: {name}\nAlready connected to: {', '.join(sorted(neighbours)) or '(nothing yet)'}",
        EXPLORE_SYSTEM,
        EXPLORE_SCHEMA,
    )
    added = 0
    async with SessionLocal() as db:
        for c in out.get("concepts") or []:
            cname = " ".join(str(c.get("name") or "").split())[:200]
            ckey = key_of(cname)
            pred = str(c.get("predicate") or "").strip().lower()[:128]
            if not ckey or ckey == key or not pred:
                continue
            # A concept that already exists -- from a document or an earlier
            # exploration -- is linked to, not duplicated: that is how separate
            # trees grow into one another.
            if ckey not in existing:
                db.add(
                    TopicNode(
                        owner_id=owner_id,
                        key=ckey,
                        name=cname,
                        type=c.get("type") if c.get("type") in TYPES else "concept",
                        category=c.get("category") if c.get("category") in CATEGORIES else "Other",
                        origin="explored",
                    )
                )
                existing[ckey] = cname
            db.add(TopicEdge(owner_id=owner_id, source_key=key, target_key=ckey, predicate=pred))
            added += 1
        await db.commit()
    log.info("topic_explored", key=key, added=added)
    return {"key": key, "added": added}


async def remove_topic(owner_id: str | None, key: str) -> None:
    async with SessionLocal() as db:
        await db.execute(
            delete(TopicNode).where(_owner(TopicNode.owner_id, owner_id), TopicNode.key == key)
        )
        await db.execute(
            delete(TopicEdge).where(
                _owner(TopicEdge.owner_id, owner_id),
                or_(TopicEdge.source_key == key, TopicEdge.target_key == key),
            )
        )
        await db.commit()
