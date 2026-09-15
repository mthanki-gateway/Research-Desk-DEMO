"""Metadata tools: facts ABOUT the data rather than facts found IN it.

The interesting part is not the SQL, it is the contract around the results.
They are not passages, so they cannot be cited, so they cannot travel as
`evidence` -- and every node downstream assumes "no evidence" means "nothing
to say". These pin the seams where that assumption used to hold.
"""

import pytest

from app.agent import nodes, tools
from app.services import corpus


class TestRendering:
    """Rendered as lines, not JSON: the model rewrites lines and echoes JSON."""

    def test_an_empty_library_says_so_plainly(self):
        out = corpus.render_documents([], [])
        assert "No documents" in out
        # Must not imply a search failed. Nothing was searched.
        assert "search" not in out.lower() or "nothing indexed" in out.lower()

    def test_a_document_that_cannot_be_searched_is_flagged_loudly(self):
        """The most useful thing this tool can report.

        A document still indexing or failed is invisible to every search, and
        "why can't you find anything in my report?" is unanswerable without
        it -- the answer is usually that the report never finished indexing.
        """
        out = corpus.render_documents(
            [
                {
                    "filename": "q1.md",
                    "status": "failed",
                    "n_chunks": 0,
                    "n_pages": 0,
                    "n_chars": 0,
                    "uploaded": "2026-01-01",
                }
            ],
            [],
        )
        assert "FAILED" in out
        assert "not searchable" in out

    def test_headings_are_offered_as_what_the_documents_cover(self):
        out = corpus.render_documents(
            [
                {
                    "filename": "a.md",
                    "status": "ready",
                    "n_chunks": 3,
                    "n_pages": 1,
                    "n_chars": 900,
                    "uploaded": None,
                }
            ],
            ["## Revenue", "## Risks"],
        )
        assert "## Revenue" in out and "## Risks" in out

    def test_nested_stats_are_inlined_not_dumped(self):
        out = corpus.render_stats("Corpus:", {"n_documents": 5, "by_status": {"ready": 5}})
        assert "n documents: 5" in out
        assert "ready: 5" in out
        assert "{" not in out  # no JSON leaking into the observation


class TestNonRetrievalRouting:
    """Which tools count as "the turn searched for something"."""

    def test_metadata_tools_are_not_retrieval(self):
        """`react` keys control flow off this set. A metadata call leaves the
        evidence list empty, so counting it as a search would send the turn to
        `draft` with nothing to compose from."""
        for name in (
            tools.LIST_DOCUMENTS,
            tools.CORPUS_STATS,
            tools.CONVERSATION_STATS,
            tools.REMEMBER_PREFERENCE,
        ):
            assert name in tools.NON_RETRIEVAL

    def test_the_search_tools_are(self):
        for name in (tools.SEARCH_DOCUMENTS, tools.SEARCH_WEB, tools.READ_AROUND):
            assert name not in tools.NON_RETRIEVAL


@pytest.mark.asyncio
class TestDispatch:
    async def test_a_metadata_call_records_the_fact_and_returns_no_hits(
        self, monkeypatch
    ):
        """The fact has to survive to whichever node writes the answer.

        On the retrieval path the agent's own prose is discarded, so a turn
        that both searched and counted would otherwise keep the passages and
        silently lose the count.
        """
        async def fake_stats(owner_id):
            assert owner_id == "tenant-a"
            return {"n_documents": 2}

        monkeypatch.setattr(corpus, "corpus_stats", fake_stats)
        facts: list[str] = []
        hits, observation = await tools.run_tool(
            tools.CORPUS_STATS,
            {},
            top_k=5,
            document_ids=None,
            owner_id="tenant-a",
            facts=facts,
        )
        assert hits == []  # never evidence
        assert facts == [observation]
        assert "n documents: 2" in observation

    async def test_a_failing_metadata_tool_does_not_end_the_loop(self, monkeypatch):
        """Same contract as the search tools: failures come back as text the
        model can react to, never as an exception."""

        async def boom(owner_id):
            raise RuntimeError("database is on fire")

        monkeypatch.setattr(corpus, "conversation_stats", boom)
        hits, observation = await tools.run_tool(
            tools.CONVERSATION_STATS, {}, top_k=5, document_ids=None, owner_id=None
        )
        assert hits == []
        assert "failed" in observation.lower()

    async def test_metadata_tools_do_not_need_a_query(self, monkeypatch):
        """They are dispatched BEFORE the query guard. Falling through would
        reject every call with "the query parameter was empty"."""

        async def fake_docs(owner_id):
            return []

        monkeypatch.setattr(corpus, "documents", fake_docs)
        _, observation = await tools.run_tool(
            tools.LIST_DOCUMENTS, {}, top_k=5, document_ids=None, owner_id=None
        )
        assert "query parameter" not in observation


class TestFactsAreNotSources:
    """The contract that keeps a computed figure from growing a citation."""

    def test_the_block_forbids_a_citation_marker(self):
        block = nodes._facts_block({"corpus_facts": ["n documents: 5"]})
        assert "NEVER put a [n] citation" in block
        assert "n documents: 5" in block

    def test_no_facts_adds_nothing_to_the_prompt(self):
        """Not a placeholder. A line saying there are no statistics is itself
        something the model reasons about."""
        assert nodes._facts_block({}) == ""
        assert nodes._facts_block({"corpus_facts": []}) == ""


@pytest.mark.asyncio
class TestDraftWithoutPassages:
    async def test_facts_alone_are_enough_to_answer(self, monkeypatch):
        """"No passages" is not "nothing to say".

        "You have 5 documents, 39 passages" is a complete answer backed by no
        passage at all. Bailing out on an empty evidence list replied "nothing
        was found" to a question the app had already answered exactly.
        """
        called = {}

        class _Pool:
            async def generate(self, prompt, **kw):
                called["prompt"] = prompt
                return '{"answer": "You have 5 documents.", "sources_used": []}'

        monkeypatch.setattr(nodes, "get_pool", lambda _role: _Pool())
        out = await nodes.draft(
            {
                "question": "how many documents do I have?",
                "evidence": [],
                "corpus_facts": ["n documents: 5"],
            }
        )
        assert "Nothing was found" not in out["draft"]
        # The bare "Sources:" heading must not appear with nothing under it --
        # the drafter opens by apologising for an empty search directly above
        # the figures that answer the question.
        assert "Sources:" not in called["prompt"]
        assert "n documents: 5" in called["prompt"]

    async def test_no_evidence_and_no_facts_still_bails_out(self):
        out = await nodes.draft({"question": "x", "evidence": [], "corpus_facts": []})
        assert "Nothing was found" in out["draft"]
