"""Ingest the evaluation corpus directly, bypassing HTTP and auth.

    docker compose exec api python -m app.scripts.ingest_fixtures --owner <uuid>
    docker compose exec api python -m app.scripts.ingest_fixtures --list

Why this exists: the golden set has to be evaluated against a *known* corpus,
and uploading fixtures through `POST /documents` needs a Supabase access token.
Getting one into a container shell for a repeatable setup step is friction with
no payoff, so this calls the ingestion service directly.

`--owner` matters. With auth enabled, documents carry an `owner_id` and
retrieval filters on it, so fixtures ingested with the wrong owner are
invisible to evaluation — which looks exactly like retrieval failing. Pass the
Supabase user id you sign in with, or omit it to ingest as the anonymous
(auth-disabled) user.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from app.db.models import DocStatus, Document
from app.db.session import SessionLocal, create_tables
from app.services.ingest import ingest_document


# Same upward search as golden.py, for the same reason: the directory depth
# differs between the host checkout and the container.
def _fixtures_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "fixtures"
        if (candidate / "golden.yaml").is_file():
            return candidate
    return Path("/app/fixtures")


# Documents that make up the evaluation corpus. `README.md` is excluded: it
# documents the fixtures rather than being one, and indexing it would let
# retrieval answer questions from the instructions instead of the corpus.
EXCLUDE = {"README.md", "golden.yaml"}


async def _existing() -> dict[str, str]:
    async with SessionLocal() as db:
        rows = (await db.execute(select(Document.filename, Document.status))).all()
    return {name: str(status) for name, status in rows}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=None, help="Supabase user id to own the documents")
    parser.add_argument("--list", action="store_true", help="show what is already ingested")
    parser.add_argument("--force", action="store_true", help="re-ingest files already present")
    args = parser.parse_args()

    await create_tables()
    fixtures = _fixtures_dir()
    present = await _existing()

    if args.list:
        print(f"fixtures dir: {fixtures}")
        print(f"{'file':44s} status")
        for path in sorted(fixtures.glob("*.md")):
            if path.name in EXCLUDE:
                continue
            print(f"{path.name:44s} {present.get(path.name, '-- not ingested --')}")
        return 0

    candidates = [p for p in sorted(fixtures.glob("*.md")) if p.name not in EXCLUDE]
    if not candidates:
        print(f"no .md fixtures found in {fixtures}", file=sys.stderr)
        return 1

    ingested = 0
    for path in candidates:
        if path.name in present and not args.force:
            print(f"skip   {path.name}  (already present: {present[path.name]})")
            continue
        data = path.read_bytes()
        print(f"ingest {path.name}  ({len(data) / 1024:.1f} KB) ...", flush=True)

        # Two steps, mirroring what POST /documents does: create the row, then
        # run ingestion against its id. `ingest_document` takes a document_id
        # rather than a filename because it is normally a background task.
        # text/markdown so the chunker takes the heading-aware path.
        async with SessionLocal() as db:
            doc = Document(
                filename=path.name,
                content_type="text/markdown",
                size_bytes=len(data),
                status=DocStatus.pending,
                owner_id=args.owner,
            )
            db.add(doc)
            await db.commit()
            await db.refresh(doc)
            doc_id = doc.id

        # Awaited, not backgrounded: this is a setup script, so it should not
        # exit before embedding finishes. Embedding is rate-limited to roughly
        # 133 chunks/minute, so a large corpus takes real minutes.
        await ingest_document(doc_id, data)
        ingested += 1

    print(f"\ningested {ingested} document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
