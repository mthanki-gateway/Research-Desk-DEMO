"""Where the original files live.

THE PROBLEM THIS SOLVES

Uploads were parsed in memory and thrown away. That is fine right up until the
first time chunking improves -- and then every document in the corpus has to be
re-uploaded by hand, because the only copy of a PDF was the text extracted from
it under last month's rules. Keeping the original means re-ingesting is a
background job rather than an apology.

It also makes a citation checkable. "Page 4 of the handbook" is a claim until
somebody can open page 4.

WHICH PROVIDER

Any of them. The backend is S3-compatible and the ENDPOINT IS A SETTING, which
is the whole point: every option worth having speaks the S3 API, so the choice
becomes a URL rather than a code change.

    Cloudflare R2   10GB, zero egress   -- needs a card
    Supabase        1GB, 5GB egress     -- NO CARD, and already a dependency here
    Backblaze B2    10GB                -- card for verification
    Filebase        5GB                 -- no card
    MinIO           self-hosted

Cloudflare R2 was the original pick, for one number: zero egress, on a store
whose entire job is serving files back. It turned out to require a credit card
to enable, which is a hard stop for a project that should be runnable by
anyone.

SUPABASE IS THE DEFAULT RECOMMENDATION for that reason, and because this app
already uses Supabase for auth -- no new account, no new vendor, and its
storage speaks S3 through `https://<project>.supabase.co/storage/v1/s3`.

Its 1GB is the real constraint, and it is worth knowing before it bites:
documents are small, but a ten-minute interview recording is about 19MB, so
audio fills that quota in roughly fifty interviews. Postgres `bytea` was
rejected outright -- it bloats every backup and burns Neon's quota.

TWO BACKENDS, ONE INTERFACE

`LocalStorage` writes to a Docker volume and is the default, so the whole
feature works with no account, no keys and no network -- which is also what
makes it testable here. `S3Storage` is the same interface over boto3, pointed
at whichever endpoint is configured. Choosing between them is one setting, and
nothing above this module knows which is in use.

TWO RULES, AGREED UP FRONT

  Store the ORIGINAL, never the extracted text. Re-chunking needs the source.
  Key by UUID, never by filename. Filenames collide, and a filename in a path
  is how directory traversal gets in.
  Prefix by TENANT -- `t/<owner>/docs/<uuid>.pdf`. See `key_for` for why that
  is worth doing even though it is not what keeps tenants apart.
"""

from __future__ import annotations

import asyncio
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import structlog

from app.config import get_settings

log = structlog.get_logger()


class StorageError(RuntimeError):
    """A file could not be stored or fetched."""


class Storage(ABC):
    """Put bytes somewhere, get them back, or hand out a link to them."""

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str) -> str:
        """Store `data` under `key`. Returns the key."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """The bytes back. Raises `StorageError` when the key is unknown."""

    @abstractmethod
    async def signed_url(self, key: str, ttl: int = 3600) -> str | None:
        """A time-limited URL, or None when this backend cannot mint one."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove it. Never raises for a key that is already gone."""


class LocalStorage(Storage):
    """A directory on disk, mounted as a volume.

    The DEFAULT, deliberately. Object storage should not be the thing standing
    between somebody cloning this repo and seeing it work, and a backend that
    only runs when a paid account is configured is a backend nobody tests.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # RESOLVED AND CHECKED, not merely joined. `key` reaches this from a
        # database row, and a row that ever contained "../../etc/passwd" would
        # otherwise be read straight off the host. Keys are minted from UUIDs,
        # so this should be impossible -- which is exactly the kind of
        # assumption worth enforcing rather than trusting.
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise StorageError(f"Refusing a key that escapes the store: {key!r}")
        return path

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        return key

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise StorageError(f"No stored file for {key!r}")
        return await asyncio.to_thread(path.read_bytes)

    async def signed_url(self, key: str, ttl: int = 3600) -> str | None:
        # None on purpose: there is no public host to sign for. The API serves
        # these bytes itself, and the caller falls back to that route.
        return None

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(shutil.rmtree, path, True) if path.is_dir() else None
        if path.is_file():
            await asyncio.to_thread(path.unlink, True)


class S3Storage(Storage):
    """Any S3-compatible object store, chosen by endpoint.

    R2, Supabase, Backblaze, Filebase and a self-hosted MinIO are all this
    class with a different URL. There is deliberately no per-provider subclass:
    the differences between them are a hostname and a quota, and a class each
    would be five copies of the same six calls waiting to drift apart.

    boto3 is synchronous, so every call goes through a thread. That is the
    right trade here -- these are a handful of calls per upload, not a hot
    path, and an async S3 client is another dependency to keep current.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "auto",
        public_base: str = "",
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - depends on install extra
            raise StorageError(
                "S3 storage needs boto3. Install the `storage` extra, or set "
                "STORAGE_BACKEND=local."
            ) from exc

        if not endpoint_url:
            raise StorageError(
                "STORAGE_BACKEND=s3 needs S3_ENDPOINT_URL. For Supabase that is "
                "https://<project>.supabase.co/storage/v1/s3; for R2, "
                "https://<account>.r2.cloudflarestorage.com."
            )

        self.bucket = bucket
        self.public_base = public_base.rstrip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            # R2 ignores the region but the SDK insists on one, and "auto" is
            # what Cloudflare's own documentation uses. Supabase wants its
            # project region, so this is a setting rather than a constant.
            region_name=region or "auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    async def get(self, key: str) -> bytes:
        try:
            response = await asyncio.to_thread(
                self._client.get_object, Bucket=self.bucket, Key=key
            )
        except Exception as exc:  # noqa: BLE001 - botocore's errors are dynamic
            raise StorageError(f"No stored file for {key!r}") from exc
        return await asyncio.to_thread(response["Body"].read)

    async def signed_url(self, key: str, ttl: int = 3600) -> str | None:
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl,
        )

    async def delete(self, key: str) -> None:
        try:
            await asyncio.to_thread(
                self._client.delete_object, Bucket=self.bucket, Key=key
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("storage_delete_failed", key=key, error=str(exc)[:200])


_store: Storage | None = None


def get_storage() -> Storage:
    """The configured backend, built once."""
    global _store
    if _store is not None:
        return _store

    settings = get_settings()
    # "r2" is accepted as well, because that is what the original design
    # called it and a setting that silently stops working is a bad trade for a
    # tidier name.
    if settings.storage_backend in ("s3", "r2"):
        _store = S3Storage(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            public_base=settings.s3_public_base,
        )
        log.info(
            "storage_ready",
            backend="s3",
            bucket=settings.s3_bucket,
            endpoint=settings.s3_endpoint_url,
        )
    else:
        _store = LocalStorage(settings.storage_dir)
        log.info("storage_ready", backend="local", root=settings.storage_dir)
    return _store


def reset_storage() -> None:
    """Drop the cached backend. For tests, and for a settings change."""
    global _store
    _store = None


# The segment an anonymous upload lives under.
#
# `owner_id` is None when auth is off, and "None" or "" as a path segment is
# how one tenant's prefix quietly becomes another's. A named sentinel makes
# those files findable and deletable like anybody else's, and it mirrors the
# `COALESCE(owner_id, '')` the SQL side already uses for the same reason.
ANONYMOUS_TENANT = "_local"


def _tenant(owner_id: str | None) -> str:
    """One path segment for an owner, safe to concatenate.

    Owner ids are Supabase UUIDs, so this should never change anything --
    which is exactly why it is here. A key is built once and stored for ever;
    an id that ever contained a slash would silently reparent every file that
    user uploads, and nothing downstream would notice.
    """
    raw = (owner_id or "").strip()
    if not raw:
        return ANONYMOUS_TENANT
    safe = "".join(c for c in raw if c.isalnum() or c in "-_")[:64]
    return safe or ANONYMOUS_TENANT


def key_for(owner_id: str | None, document_id, filename: str) -> str:
    """`t/<owner>/docs/<uuid><ext>` -- keyed by TENANT, then by id.

    WHY THE OWNER IS IN THE PATH

    Not because it is what keeps tenants apart -- the API does that, and it
    checks ownership before it ever looks at a key. The prefix earns its place
    three other ways:

      DELETING A TENANT becomes one prefixed list instead of a full scan
      joined against the database. So does measuring what one is using, and so
      does exporting everything they own when they ask for it.

      BUCKET POLICIES can be scoped by prefix -- S3 IAM and Supabase Storage
      rules both work on paths. A flat namespace cannot be divided at all, so
      every credential is necessarily a credential for everything.

      DEFENCE IN DEPTH. Ownership is enforced in one place today. A prefix
      means a future mistake there is not automatically a cross-tenant read.

    The EXTENSION ONLY from the filename, never the name. A filename in a key
    is a collision waiting to happen and a traversal waiting to be tried; the
    extension is kept because it is what makes a signed URL open in a viewer
    rather than download as a blob.

    Old keys keep working. `storage_key` is STORED per document rather than
    derived, so anything written under the previous flat scheme is still
    fetched from where it actually is -- changing this function does not
    orphan a single file.
    """
    suffix = Path(filename or "").suffix.lower()[:12]
    if not suffix.isascii() or any(c in suffix for c in "/\.." if c != "."):
        suffix = ""
    return f"t/{_tenant(owner_id)}/docs/{document_id}{suffix}"


def media_key_for(owner_id: str | None, session_id, turn: int, ext: str) -> str:
    """`t/<owner>/howl/<session>/<n>.<ext>` -- for interview recordings.

    The same tenant prefix, so one delete removes a tenant's documents and
    their recordings together.

    THE OWNER IS THE PROJECT'S OWNER, not the person speaking. A guest holding
    a magic link has no account and owns nothing; the recording belongs to
    whoever sent the link, which is the same answer the conversation row
    already gives.
    """
    clean = "".join(c for c in (ext or "wav") if c.isalnum())[:8] or "wav"
    return f"t/{_tenant(owner_id)}/howl/{session_id}/{int(turn):04d}.{clean}"
