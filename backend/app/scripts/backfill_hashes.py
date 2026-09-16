"""Stamp a content hash onto documents ingested before the column existed.

WHY THIS IS NEEDED

Dedupe matches on a SHA-256 of the uploaded bytes, and those bytes are not
retained after ingestion -- only the parsed chunks are. So every document
already in the library has a NULL hash and matches nothing: re-uploading it
creates a duplicate, exactly as before.

The chunk text cannot be used to close the gap. It is the PARSED form, and
re-joining it would not reproduce the original file byte for byte -- different
bytes, different hash, no match. There is no way to recover a hash from the
database alone.

What CAN close the gap is the original files, if you still have them. This
walks a directory, hashes each file, and stamps the hash onto the document row
with that filename. It is a filename match, which is the weak heuristic the
upload path deliberately avoids -- but here it is safe, because a human is
asserting "these files are that library" by choosing the directory.

    docker compose exec api python -m app.scripts.backfill_hashes /app/fixtures

Dry by default. Add --write to commit.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import pathlib

from sqlalchemy import select

from app.db.models import Document
from app.db.session import SessionLocal

# Matches the upload path's accepted types.
SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _digests(directory: pathlib.Path) -> dict[str, str]:
    """Hash every uploadable file under `directory`, keyed by filename.

    All the disk work happens here, in one synchronous pass, so the async
    function below never blocks the event loop on IO -- hashing a directory of
    PDFs is not free.

    A filename appearing twice under different subdirectories keeps the first;
    there is no way to tell which one the library meant, and guessing would
    stamp the wrong hash onto a document silently.
    """
    out: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        out.setdefault(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    return out


async def backfill(directory: pathlib.Path, write: bool) -> int:
    files = await asyncio.to_thread(_digests, directory)
    if not files:
        print(f"No uploadable files under {directory}")
        return 0

    stamped = 0
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Document).where(Document.content_sha256.is_(None))
            )
        ).scalars().all()

        if not rows:
            print("Every document already has a hash. Nothing to do.")
            return 0

        for doc in rows:
            digest = files.get(doc.filename)
            if digest is None:
                print(f"  skip   {doc.filename}  (no file of that name on disk)")
                continue

            # Another row may already hold this hash -- the library can contain
            # two copies of one file, which is the very thing that prompted
            # this. Stamping both would violate the unique index and abort the
            # whole transaction, so the duplicate is reported and left alone
            # for a human to delete.
            clash = (
                await session.execute(
                    select(Document).where(
                        Document.content_sha256 == digest,
                        Document.owner_id.is_(None)
                        if doc.owner_id is None
                        else Document.owner_id == doc.owner_id,
                    )
                )
            ).scalars().first()
            if clash is not None:
                print(
                    f"  DUP    {doc.filename}  is the same content as "
                    f"{clash.filename} ({clash.id}) -- left unstamped, delete one"
                )
                continue

            doc.content_sha256 = digest
            stamped += 1
            print(f"  stamp  {doc.filename}  {digest[:12]}")

        if write:
            await session.commit()
            print(f"\nCommitted. {stamped} document(s) stamped.")
        else:
            await session.rollback()
            print(f"\nDry run. {stamped} document(s) WOULD be stamped. Re-run with --write.")

    return stamped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--write", action="store_true", help="commit the changes")
    args = parser.parse_args()
    asyncio.run(backfill(args.directory, args.write))


if __name__ == "__main__":
    main()
