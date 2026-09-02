"""Drop the Qdrant collection and clear document rows.

Needed whenever EMBEDDING_DIM or the embedding provider changes: Qdrant
collections are fixed-dimension, so existing vectors become unusable.

    docker compose exec api python -m app.scripts.reset_vectors
"""

import asyncio

from sqlalchemy import delete

from app.db.models import Chunk, Document
from app.db.session import SessionLocal, create_tables
from app.services.vectorstore import get_vector_store


async def main() -> None:
    store = get_vector_store()
    try:
        await store.drop_collection()
        print("dropped Qdrant collection")
    except Exception as exc:
        print(f"collection not dropped ({exc}) -- continuing")

    await create_tables()
    async with SessionLocal() as session:
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        await session.commit()
    print("cleared documents and chunks")

    await store.ensure_collection()
    print("recreated collection -- re-upload your documents")
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
