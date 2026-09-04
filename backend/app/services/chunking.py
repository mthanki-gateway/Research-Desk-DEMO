"""Recursive character chunking, hand-rolled.

Written out rather than pulled from langchain-text-splitters because chunking
is the single biggest lever on retrieval quality, and it's worth being able to
see and tune the algorithm.

The idea: try to split on the most semantically meaningful boundary that yields
pieces under the size limit -- paragraphs first, then lines, then sentences,
then words, and only mid-word as a last resort. Splitting mid-sentence strands
the subject of a claim in one chunk and its number in another, which is exactly
the failure that makes a RAG answer confidently wrong.
"""

import re
from dataclasses import dataclass

from app.services.parsing import Page

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class TextChunk:
    index: int
    page: int | None
    text: str
    heading: str | None = None


def chunk_pages(
    pages: list[Page], *, size: int, overlap: int, min_chars: int = 0
) -> list[TextChunk]:
    """Chunk each page independently so a chunk never straddles two pages.

    That costs a little efficiency at page boundaries but keeps every chunk
    attributable to exactly one page, which is what makes citations honest.

    Within a page, markdown headings are treated as HARD boundaries. Without
    this, a 900-char window happily swallows the tail of one section plus the
    head of the next, and the blended embedding sits between both topics --
    near neither. Measured on the test report: asking for "main risks"
    returned the headcount section until sections were split here.
    """
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    candidates: list[TextChunk] = []
    for page in pages:
        for heading, body in _split_sections(page.text):
            for piece in _split(body, size, overlap):
                # Prepend the heading so an isolated chunk still states its own
                # context. Retrieval sees "## Risks" next to the risk text,
                # which measurably sharpened the match.
                already_prefixed = heading is None or piece.startswith(heading)
                text = piece if already_prefixed else f"{heading}\n{piece}"
                candidates.append(
                    TextChunk(
                        index=0,  # assigned after filtering, so they stay dense
                        page=page.number,
                        text=text,
                        heading=heading,
                    )
                )

    kept = _drop_contentless(candidates, min_chars=min_chars)
    for i, chunk in enumerate(kept):
        chunk.index = i
    return kept


def _body_length(chunk: TextChunk) -> int:
    """Length of a chunk MINUS its heading prefix.

    Measuring the whole text would be wrong: every chunk carries its heading, so
    a long heading can push a content-free chunk over any threshold. What
    matters is how much actual body there is underneath it.
    """
    text = chunk.text
    if chunk.heading and text.startswith(chunk.heading):
        text = text[len(chunk.heading) :]
    return len(text.strip())


def _drop_contentless(chunks: list[TextChunk], *, min_chars: int) -> list[TextChunk]:
    """Remove chunks with too little body to answer anything.

    The case that motivated this: `_split_sections` deliberately keeps a heading
    that has no body of its own (an H1 title immediately followed by an H2) so
    the title stays searchable. In practice that produced a 39-character chunk
    of pure heading -- "Acme Corporation — Annual Report 2024" -- which then
    ranked SECOND in every retrieval, because a bare title embeds close to
    almost any question about the document. It consumed one of five slots and
    contributed nothing any answer could cite.

    The document title is not lost: it stays in `documents.filename`, it is
    shown in every citation, and it is prefixed onto real chunks in its own
    section.

    Guard: if filtering would empty the document, keep everything. A one-line
    file is a legitimate upload, and returning zero chunks would fail ingestion
    for a document that is merely short rather than broken.
    """
    if min_chars <= 0:
        return chunks

    kept = [c for c in chunks if _body_length(c) >= min_chars]
    return kept if kept else chunks


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown into (heading, body) sections.

    Text with no headings -- most PDF extractions -- comes back as a single
    (None, text) section, so behaviour there is unchanged.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    # A heading with no body of its own -- an H1 document title immediately
    # followed by an H2 -- is CARRIED FORWARD and prefixed onto the next
    # heading, producing a heading path like:
    #
    #     # Incident Post-Mortem: INC-2024-1183
    #     ## Summary
    #
    # Two earlier approaches were both wrong, and the second was worse:
    #
    #   1. Emitting it as its own section produced a content-free chunk (39
    #      chars of pure title) that ranked SECOND in every retrieval, because
    #      a bare title embeds close to almost any question about the document.
    #   2. Dropping it via the min-body filter DESTROYED the text. Measured: the
    #      only occurrence of "INC-2024-1183" in the corpus was that title, so
    #      no retriever -- dense or BM25 -- could find the document by its own
    #      incident number. The label validator caught it; a benchmark had
    #      already mis-attributed the loss to "dense retrieval is bad at
    #      identifiers", which sent the diagnosis in entirely the wrong
    #      direction.
    #
    # Carrying it forward keeps the text searchable, attaches it to real
    # content, and creates no content-free chunk. It is also just what a
    # heading path means.
    pending: list[str] = []
    for i, match in enumerate(matches):
        heading_line = match.group(0).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()

        if not body:
            pending.append(heading_line)
            continue

        full_heading = "\n".join([*pending, heading_line])
        pending.clear()
        sections.append((full_heading, body))

    # Trailing heading with nothing after it at all. Keep it rather than lose
    # the text -- the min-body filter will drop it only if it is genuinely too
    # short to matter, and by then there is nothing to attach it to.
    if pending:
        tail = "\n".join(pending)
        sections.append((tail, tail))
    return sections


def _split(text: str, size: int, overlap: int) -> list[str]:
    pieces = _recursive_split(text, size, SEPARATORS)
    return _merge_with_overlap(pieces, size, overlap)


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into fragments that are each <= size, or unsplittable."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        # Last resort: hard cut. Only reached by things like a single
        # enormous unbroken token (base64 blob, minified data).
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(separator)
    out: list[str] = []
    for i, part in enumerate(parts):
        # Put the separator back, except after the final part, so the text
        # round-trips and sentence punctuation survives.
        restored = part + separator if i < len(parts) - 1 else part
        if len(restored) <= size:
            if restored.strip():
                out.append(restored)
        else:
            out.extend(_recursive_split(restored, size, remaining))
    return out


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack fragments up to `size`, repeating a tail as the overlap.

    Overlap matters because a claim can span a boundary; repeating the tail of
    the previous chunk gives the next chunk enough context to stand alone.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + len(piece) <= size:
            current += piece
            continue

        chunks.append(current.strip())
        tail = _tail(current, overlap)
        current = tail + piece if len(tail) + len(piece) <= size else piece

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap chars, snapped forward to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
