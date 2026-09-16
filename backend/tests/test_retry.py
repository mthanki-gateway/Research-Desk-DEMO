"""The retry that could not possibly succeed.

WHAT WENT WRONG

Asked for the heights of the Great Pyramid and the Red Pyramid, the agent
answered: "Your documents do not give the height of the Red Pyramid." True, and
useless -- the web gives it as the first result for "Red Pyramid height".

The trace showed the machinery working correctly right up to the last step. The
critic spotted the gap and named it. `retrieve_node` then re-searched THE SAME
CORPUS, got nothing, and the turn concluded the fact was unavailable.

Two defects, both here:

1. Re-asking a corpus that has already been asked is the one retry that cannot
   work. The critic raised the gap PRECISELY BECAUSE the corpus did not answer
   it, so no rewording can help. Only a different source can.

2. The critic's gap is a sentence written to be READ -- "A height for the Red
   Pyramid from your documents" -- and it was being embedded verbatim. Half of
   it describes the shape of the omission, and that half dominates the vector.
"""

import pytest

from app.agent import nodes


class TestGapToQuery:
    """A description of an omission is not a search for the thing omitted."""

    def test_corpus_references_are_stripped(self):
        out = nodes.as_query("A height for the Red Pyramid from your documents")
        assert "documents" not in out.lower()
        # The SUBJECT has to survive. Stripping it would be worse than nothing.
        assert "Red Pyramid" in out
        assert "height" in out.lower()

    def test_a_long_gap_keeps_its_nouns(self):
        out = nodes.as_query(
            "A separate height for the Great Pyramid of Khufu distinct from "
            "the Great Pyramid of Giza in your documents"
        )
        assert "documents" not in out.lower()
        assert "Khufu" in out and "Giza" in out

    @pytest.mark.parametrize(
        "phrase",
        [
            "in your documents",
            "from your documents",
            "in the user's documents",
            "in the provided documents",
            "within the corpus",
            "according to the sources",
            "in these files",
        ],
    )
    def test_every_spelling_of_the_same_meta_phrase(self, phrase):
        out = nodes.as_query(f"The height of the Red Pyramid {phrase}")
        assert "Red Pyramid" in out
        for word in ("document", "corpus", "source", "file"):
            assert word not in out.lower(), (phrase, out)

    def test_a_leading_article_goes(self):
        assert not nodes.as_query("The original height of the pyramid").lower().startswith(
            "the "
        )

    @pytest.mark.parametrize(
        "query",
        [
            "Red Pyramid height",
            "connection pool sizing policy",
            "on-call paging runbook requirements",
        ],
    )
    def test_a_good_query_is_left_alone(self, query):
        """Over-trimming is as bad as not trimming.

        Most gaps are already reasonable queries, and the common case must come
        through unharmed.
        """
        assert nodes.as_query(query) == query

    def test_pure_meta_language_falls_back_to_the_original(self):
        """Stripping everything would send an empty query to the embedder."""
        assert nodes.as_query("in your documents").strip()


@pytest.mark.asyncio
class TestTheRetryReachesTheWeb:
    """The fix. A retry must be able to look somewhere new."""

    async def _run(self, monkeypatch, *, seen, web_enabled=True, document_ids=None):
        calls = {"documents": [], "web": []}

        async def fake_retrieve(query, **kwargs):
            calls["documents"].append(query)
            return []

        async def fake_search_web(query, **kwargs):
            calls["web"].append(query)
            return []

        monkeypatch.setattr(nodes, "retrieve", fake_retrieve)
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: web_enabled)
        monkeypatch.setattr(nodes.websearch, "search_web", fake_search_web)

        state = {
            "question": "q",
            "pending_queries": ["Red Pyramid height"],
            "seen_chunk_ids": seen,
            "owner_id": None,
        }
        if document_ids:
            state["document_ids"] = document_ids
        await nodes.retrieve_node(state)
        return calls

    async def test_a_retry_also_searches_the_web(self, monkeypatch):
        calls = await self._run(monkeypatch, seen={"chunk-1", "chunk-2"})
        assert calls["documents"], "stopped searching documents"
        assert calls["web"] == ["Red Pyramid height"], "the web was never retried"

    async def test_the_first_pass_does_not(self, monkeypatch):
        """ReAct already has `search_web` as a tool and chooses for itself.

        Duplicating it here would double every web call on every turn.
        """
        calls = await self._run(monkeypatch, seen=set())
        assert calls["documents"]
        assert calls["web"] == []

    async def test_a_scoped_question_stays_scoped(self, monkeypatch):
        """"Search only THESE documents" must not quietly reach the internet."""
        calls = await self._run(
            monkeypatch,
            seen={"chunk-1"},
            document_ids=["6f1a5b3c-0000-4000-8000-000000000000"],
        )
        assert calls["web"] == []

    async def test_an_unconfigured_web_is_not_called(self, monkeypatch):
        calls = await self._run(monkeypatch, seen={"chunk-1"}, web_enabled=False)
        assert calls["web"] == []

    async def test_a_failing_web_search_does_not_fail_the_turn(self, monkeypatch):
        """The document results are still worth drafting from."""

        async def fake_retrieve(query, **kwargs):
            return []

        async def boom(query, **kwargs):
            raise RuntimeError("serper is down")

        monkeypatch.setattr(nodes, "retrieve", fake_retrieve)
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: True)
        monkeypatch.setattr(nodes.websearch, "search_web", boom)

        out = await nodes.retrieve_node(
            {
                "question": "q",
                "pending_queries": ["Red Pyramid height"],
                "seen_chunk_ids": {"chunk-1"},
                "owner_id": None,
            }
        )
        assert "evidence" in out

    async def test_the_query_is_cleaned_before_it_is_searched(self, monkeypatch):
        """Both halves of the fix, at the seam where they meet."""
        calls = await self._run(monkeypatch, seen={"c"})
        monkeypatch.undo()

        seen_queries = []

        async def fake_retrieve(query, **kwargs):
            seen_queries.append(query)
            return []

        async def fake_search_web(query, **kwargs):
            seen_queries.append(query)
            return []

        monkeypatch.setattr(nodes, "retrieve", fake_retrieve)
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: True)
        monkeypatch.setattr(nodes.websearch, "search_web", fake_search_web)

        await nodes.retrieve_node(
            {
                "question": "q",
                "pending_queries": ["A height for the Red Pyramid from your documents"],
                "seen_chunk_ids": {"c"},
                "owner_id": None,
            }
        )
        assert seen_queries, "nothing was searched"
        for query in seen_queries:
            assert "documents" not in query.lower(), query
            assert "Red Pyramid" in query
        assert calls is not None
