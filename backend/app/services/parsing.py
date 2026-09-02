"""Extract text from an uploaded file, keeping page numbers for citations."""

import io
from dataclasses import dataclass

import structlog
from pypdf import PdfReader

log = structlog.get_logger()

SUPPORTED = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "text",
    "text/x-markdown": "text",
}

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".txt": "text", ".md": "text"}


@dataclass(slots=True)
class Page:
    """One page of extracted text. `number` is 1-based, or None for flat text."""

    number: int | None
    text: str


class UnsupportedFileType(Exception):
    pass


def detect_kind(filename: str, content_type: str | None) -> str:
    """Trust the extension over the browser-supplied content type.

    Browsers report .md as everything from text/markdown to application/octet-
    stream depending on OS, so the extension is the more reliable signal.
    """
    lowered = filename.lower()
    for ext, kind in SUPPORTED_EXTENSIONS.items():
        if lowered.endswith(ext):
            return kind
    if content_type in SUPPORTED:
        return SUPPORTED[content_type]
    raise UnsupportedFileType(
        f"Unsupported file '{filename}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def parse(data: bytes, filename: str, content_type: str | None) -> list[Page]:
    kind = detect_kind(filename, content_type)
    if kind == "pdf":
        return _parse_pdf(data)
    return _parse_text(data)


def _parse_pdf(data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single malformed page shouldn't kill the upload
            log.warning("pdf_page_failed", page=i, error=str(exc))
            continue
        text = _clean(text)
        if text:
            pages.append(Page(number=i, text=text))
    if not pages:
        raise UnsupportedFileType(
            "No extractable text found. This is likely a scanned PDF -- it would "
            "need OCR, which this demo does not do."
        )
    return pages


def _parse_text(data: bytes) -> list[Page]:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnsupportedFileType("Could not decode file as text.")

    cleaned = _clean(text)
    if not cleaned:
        raise UnsupportedFileType("File is empty.")
    return [Page(number=None, text=cleaned)]


def _clean(text: str) -> str:
    # PDF extraction commonly yields hard-wrapped lines and runs of blank
    # lines; both confuse the chunker's paragraph detection.
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line:
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks == 1:
                out.append("")
    return "\n".join(out).strip()
