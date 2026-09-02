"""Print the models this API key can actually serve, with their capabilities.

Model ids drift (gemma-3 -> gemma-4, embedding-001 -> -002), so rather than
trusting a hardcoded string, ask the API and paste a real id into .env.

    docker compose exec api python -m app.scripts.list_models
"""

import asyncio

import httpx

from app.config import get_settings

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


async def main() -> None:
    settings = get_settings()
    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is empty — set it in .env and restart the api service.")

    models: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        page_token = ""
        while True:
            params = {"key": settings.google_api_key, "pageSize": 200}
            if page_token:
                params["pageToken"] = page_token
            resp = await client.get(ENDPOINT, params=params)
            resp.raise_for_status()
            body = resp.json()
            models.extend(body.get("models", []))
            page_token = body.get("nextPageToken", "")
            if not page_token:
                break

    def supports(m: dict, method: str) -> bool:
        return method in m.get("supportedGenerationMethods", [])

    chat = sorted(m["name"] for m in models if supports(m, "generateContent"))
    embed = sorted(m["name"] for m in models if supports(m, "embedContent"))

    print(f"\n=== chat / reasoning ({len(chat)}) — pick one for LLM_MODEL ===")
    for name in chat:
        print(f"  {name}")

    print(f"\n=== embeddings ({len(embed)}) — pick one for EMBEDDING_MODEL ===")
    for name in embed:
        dims = next(
            (m.get("outputDimensionality") for m in models if m["name"] == name),
            None,
        )
        print(f"  {name}" + (f"   (native dim: {dims})" if dims else ""))
    print()


if __name__ == "__main__":
    asyncio.run(main())
