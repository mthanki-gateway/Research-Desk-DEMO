"""Phase 3: parent-child assembly and the context ladder.

Chunk size is otherwise one knob serving two opposed jobs -- retrieval wants
small and sharp, generation wants surrounding context. These pin the mechanism
that separates them, and in particular the dedupe, which is not a tidy-up: three
children of one section handed to the model as three sources is a fabricated
finding produced by plumbing.
"""

import uuid

import pytest

from app.config import Settings
from app.services.parents import _strip_overlap, expand_to_parents, parent_key
from app.services.vectorstore import SearchHit


def _hit(
    n: int,
    *,
    heading: str | None = "## Timeline",
    doc: int = 7,
    source: str = "document",
    url: str | None = None,
    text: str = "passage",
) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(int=n),
        document_id=uuid.UUID(int=doc),
        filename="incident.md",
        page=1,
        chunk_index=n,
        heading=heading,
        text=text,
        score=0.7,
        meta={},
        found_by=["q"],
        source=source,
        url=url,
    )


class TestParentKey:
    def test_same_section_same_parent(self):
        assert parent_key(_hit(1)) == parent_key(_hit(2))

    def test_different_sections_differ(self):
        assert parent_key(_hit(1, heading="## Timeline")) != parent_key(
            _hit(2, heading="## Root Cause")
        )

    def test_same_heading_in_different_documents_differ(self):
        """A "## Summary" in two files is two sections. Keying on the heading
        alone would splice unrelated documents into one parent."""
        assert parent_key(_hit(1, doc=7)) != parent_key(_hit(1, doc=8))

    def test_a_headingless_chunk_is_its_own_parent(self):
        """Most PDF extractions have no headings. Grouping every headingless
        chunk of a file together would assemble the ENTIRE document -- which is
        not a parent, it is the absence of one."""
        a, b = _hit(1, heading=None), _hit(2, heading=None)
        assert parent_key(a) != parent_key(b)

    def test_a_web_result_is_its_own_parent(self):
        """There is no document to expand within, and a url is already a whole
        page."""
        hit = _hit(1, source="web", url="https://example.com/a")
        assert parent_key(hit).startswith("web:")


class TestStripOverlap:
    def test_removes_the_repeated_tail(self):
        prev = "the timeline begins at 09:00 and the first alert fires"
        cur = "and the first alert fires at 09:04, paging the on-call engineer"
        out = _strip_overlap(prev, cur, 150)
        assert out.startswith("at 09:04")
        assert "the first alert fires at 09:04" not in out

    def test_leaves_unrelated_text_alone(self):
        assert _strip_overlap("completely different", "nothing shared here", 150) == (
            "nothing shared here"
        )

    def test_ignores_coincidental_short_repeats(self):
        """Only runs longer than the short-match floor count. Without it a
        shared " the " would be treated as the overlap and eat real text."""
        assert _strip_overlap("ends with the", "the beginning of a new part", 150) == (
            "the beginning of a new part"
        )

    def test_handles_an_empty_side(self):
        assert _strip_overlap("", "content", 150) == "content"


@pytest.mark.asyncio
class TestExpandToParents:
    async def _settings(self, monkeypatch, **overrides):
        import app.services.parents as parents_mod

        s = Settings(google_api_key="x", **overrides)
        monkeypatch.setattr(parents_mod, "get_settings", lambda: s)
        return s

    async def test_disabled_is_a_true_no_op(self, monkeypatch):
        await self._settings(monkeypatch, parent_retrieval=False)
        hits = [_hit(1), _hit(2)]
        assert await expand_to_parents(hits) is hits

    async def test_collapses_children_of_one_section(self, monkeypatch):
        """THE POINT. Relevant passages cluster, so several winners routinely
        share a section. Handing the model the same section three times makes it
        write "the report repeatedly emphasises" -- a finding fabricated by a
        plumbing bug."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True)

        async def fake_children(document_id, heading):
            return [(1, "first part."), (2, "second part.")]

        monkeypatch.setattr(parents_mod, "_fetch_children", fake_children)

        out = await expand_to_parents([_hit(1), _hit(2), _hit(3)])
        assert len(out) == 1, "three children of one section are one source"
        assert "first part." in out[0].text and "second part." in out[0].text

    async def test_keeps_distinct_sections_apart(self, monkeypatch):
        """Two sections making the same point is a REAL finding. The dedupe key
        is identity, never similarity."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True)

        async def fake_children(document_id, heading):
            return [(1, f"body of {heading}"), (2, "more")]

        monkeypatch.setattr(parents_mod, "_fetch_children", fake_children)

        out = await expand_to_parents(
            [_hit(1, heading="## Timeline"), _hit(2, heading="## Root Cause")]
        )
        assert len(out) == 2

    async def test_rank_order_is_preserved(self, monkeypatch):
        """The best child speaks for its parent. Re-sorting would discard the
        reranker's judgement, which is the most informed ordering available."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True)

        async def fake_children(document_id, heading):
            return [(1, "a"), (2, "b")]

        monkeypatch.setattr(parents_mod, "_fetch_children", fake_children)

        out = await expand_to_parents(
            [_hit(1, heading="## Second"), _hit(2, heading="## First")]
        )
        assert [h.heading for h in out] == ["## Second", "## First"]

    async def test_a_lone_child_is_returned_unchanged(self, monkeypatch):
        """The chunk IS the section. Re-using the hit keeps its text identical
        to what retrieval actually scored."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True)
        monkeypatch.setattr(
            parents_mod, "_fetch_children", lambda d, h: _one_child()
        )

        original = _hit(1, text="the only chunk")
        out = await expand_to_parents([original])
        assert out[0] is original

    async def test_truncation_is_announced_not_silent(self, monkeypatch):
        """A silent drop looks like a model failure: the answer omits something
        the section plainly contains and nothing explains why."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True, parent_max_chars=50)

        async def fake_children(document_id, heading):
            return [(i, "x" * 40) for i in range(1, 6)]

        monkeypatch.setattr(parents_mod, "_fetch_children", fake_children)

        out = await expand_to_parents([_hit(1)])
        assert "[section continues]" in out[0].text
        assert len(out[0].text) < 200

    async def test_a_fetch_failure_keeps_the_original_hit(self, monkeypatch):
        """An expansion that throws would lose retrieval that already
        succeeded. The un-expanded chunk is exactly the prior behaviour."""
        import app.services.parents as parents_mod

        await self._settings(monkeypatch, parent_retrieval=True)

        async def boom(document_id, heading):
            raise RuntimeError("postgres is down")

        monkeypatch.setattr(parents_mod, "_fetch_children", boom)

        out = await expand_to_parents([_hit(1, text="original text")])
        assert out[0].text == "original text"

    async def test_web_hits_pass_through(self, monkeypatch):
        await self._settings(monkeypatch, parent_retrieval=True)
        hit = _hit(1, source="web", url="https://example.com/a", text="snippet")
        out = await expand_to_parents([hit])
        assert out == [hit]


async def _one_child():
    return [(1, "the only chunk")]


class TestParentDefaults:
    def test_off_until_measured(self):
        """It cannot move Tier 1 -- that scores which CHUNKS were retrieved, and
        this changes the text handed to the model afterwards -- so the evidence
        has to come from a Tier 2 judge run."""
        assert Settings(google_api_key="x").parent_retrieval is False

    def test_the_cap_is_smaller_than_a_gemma_minute(self):
        """Five expanded parents must not dwarf the token budget. On the gemma
        profile a minute's entire allowance is 16K tokens, roughly 64K
        characters."""
        s = Settings(google_api_key="x")
        assert s.parent_max_chars * s.retrieval_top_k < 64_000
