"""Chunking tests.

Chunking is the highest-leverage code in the retrieval path and the easiest to
break silently, so these tests pin the two behaviours that were each got wrong
once: what happens to a heading with no body, and what counts as a chunk too
small to keep.
"""

import pytest

from app.services.chunking import _body_length, chunk_pages
from app.services.parsing import Page


def page(text: str) -> list[Page]:
    return [Page(number=None, text=text)]


class TestHeadingWithNoBody:
    """The bug that cost the corpus an incident number.

    `# Title` followed immediately by `## Section` gives the H1 no body. It
    must be carried forward onto the next heading, not emitted alone (a
    content-free chunk that ranks highly on everything) and not dropped (which
    destroys the only copy of the title text).
    """

    DOC = """# Incident Post-Mortem: INC-2024-1183 — Payments Cutover Outage

## Summary
On 14 November 2024 the payments service was unavailable for 2 hours and 47
minutes during a planned migration cutover.

## Root Cause
The immediate cause was connection pool exhaustion in the new service.
"""

    def test_title_text_survives(self):
        chunks = chunk_pages(page(self.DOC), size=900, overlap=150, min_chars=50)
        combined = "\n".join(c.text for c in chunks)
        # THE regression: this string existed only in the body-less H1.
        assert "INC-2024-1183" in combined

    def test_title_is_attached_to_the_first_real_section(self):
        chunks = chunk_pages(page(self.DOC), size=900, overlap=150, min_chars=50)
        first = chunks[0]
        assert "INC-2024-1183" in first.text
        assert "Summary" in first.text
        # Matched without "minutes": the source wraps mid-phrase, and chunking
        # preserves source newlines rather than reflowing text.
        assert "2 hours and 47" in first.text

    def test_no_content_free_chunk_is_produced(self):
        # The other failure mode: a 39-char chunk of pure title that ranked
        # second in every retrieval.
        chunks = chunk_pages(page(self.DOC), size=900, overlap=150, min_chars=50)
        assert all(_body_length(c) >= 50 for c in chunks)

    def test_later_sections_do_not_inherit_the_title(self):
        # The heading path applies to the section that consumed it, not to
        # every subsequent section -- otherwise every chunk carries the title
        # and the embeddings all drift toward it.
        chunks = chunk_pages(page(self.DOC), size=900, overlap=150, min_chars=50)
        root_cause = [c for c in chunks if "connection pool" in c.text]
        assert root_cause
        assert "INC-2024-1183" not in root_cause[0].text


class TestMinBodyFilter:
    def test_measures_body_not_total_length(self):
        """A long heading must not let a content-free chunk clear the bar."""
        doc = "## A Very Long Section Heading That Is Itself Over Fifty Characters\n\n"
        chunks = chunk_pages(page(doc), size=900, overlap=150, min_chars=50)
        # The heading is long, the body is empty. Nothing should survive on the
        # strength of the heading alone -- but the guard below keeps it rather
        # than returning zero chunks.
        assert all(_body_length(c) == 0 for c in chunks)

    def test_never_empties_a_document(self):
        """A short file is a legitimate upload, not a failed ingest."""
        chunks = chunk_pages(page("Short note."), size=900, overlap=150, min_chars=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short note."

    def test_drops_a_short_section_among_long_ones(self):
        doc = """## Real Section
This section has substantially more than fifty characters of actual body text
so that it comfortably survives the minimum-body filter.

## Stub
x
"""
        chunks = chunk_pages(page(doc), size=900, overlap=150, min_chars=50)
        texts = "\n".join(c.text for c in chunks)
        assert "substantially more" in texts
        assert "## Stub" not in texts

    def test_indices_stay_dense_after_filtering(self):
        doc = """## Stub
x

## Real One
Body text long enough to survive the minimum-body filter without any trouble.

## Real Two
Another body that is comfortably longer than the fifty character threshold.
"""
        chunks = chunk_pages(page(doc), size=900, overlap=150, min_chars=50)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class TestHeadingBoundaries:
    def test_sections_do_not_bleed_into_each_other(self):
        """The measured win: a 900-char window used to swallow the tail of one
        section plus the head of the next, and the blended embedding sat between
        both topics, near neither."""
        doc = """## Revenue
Revenue was 847.3 million dollars in fiscal 2024.

## Headcount
Total headcount stood at 4,182 employees at the end of the year.
"""
        chunks = chunk_pages(page(doc), size=900, overlap=150, min_chars=0)
        revenue = [c for c in chunks if "847.3" in c.text]
        assert revenue
        assert "4,182" not in revenue[0].text

    def test_heading_is_prefixed_into_the_chunk(self):
        doc = "## Risks\nCustomer concentration is the principal risk identified.\n"
        chunks = chunk_pages(page(doc), size=900, overlap=150, min_chars=0)
        assert chunks[0].text.startswith("## Risks")
        assert chunks[0].heading == "## Risks"

    def test_text_without_headings_still_chunks(self):
        doc = "Plain prose with no markdown headings at all. " * 40
        chunks = chunk_pages(page(doc), size=300, overlap=50, min_chars=0)
        assert len(chunks) > 1
        assert all(c.heading is None for c in chunks)


class TestOverlapGuard:
    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_pages(page("text"), size=100, overlap=100)
