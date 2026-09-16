"""Web search and the ReAct tool loop.

Three layers, separated by what each can actually prove without a network or a
model:

`_to_hit` is pure, so every shape a search API can return is checkable
directly -- including the id-stability property the whole dedupe story rests
on.

`tool_specs` and `run_tool` are checked against a stubbed retriever, because
the interesting behaviour is not "does search work" but "what does the model
SEE" -- which tools are declared, and that a failing tool comes back as
readable text rather than an exception that would end the loop.

`react` is driven by a scripted fake model. That is the only way to pin the
loop's invariants: the round cap, the per-round call cap, dedupe across rounds,
and -- the one that motivated gathering rather than answering -- that it
returns evidence and leaves composition to `draft`.
"""

import uuid

import pytest

from app.agent import react as react_mod
from app.agent import tools
from app.config import Settings
from app.services import websearch
from app.services.vectorstore import SearchHit

ORGANIC = {
    "title": "HNSW explained",
    "link": "https://example.com/hnsw",
    "snippet": "Hierarchical Navigable Small World graphs are a layered index.",
    "position": 1,
}


def _settings(**overrides) -> Settings:
    """Settings with the ambient environment pinned out.

    `serper_api_key=""` is EXPLICIT, and it has to be. Settings reads the
    process environment, so once a real SERPER_API_KEY was configured in .env
    every "unconfigured" test inherited it -- `test_off_without_a_key` failed,
    and `test_search_returns_empty_when_unconfigured` went to the live Serper
    API and searched for the string "anything". A test that changes behaviour
    depending on whether a developer has a key is not a test.
    """
    return Settings(
        **{"google_api_key": "x", "serper_api_key": "", **overrides}
    )


def _hit(n: int, *, source: str = "document") -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(int=n),
        document_id=uuid.UUID(int=99),
        filename=f"doc{n}.pdf",
        page=1,
        chunk_index=n,
        heading="## Findings",
        text=f"passage {n}",
        score=0.5,
        meta={},
        found_by=["q"],
        source=source,
    )


# --------------------------------------------------------------------------
# _to_hit
# --------------------------------------------------------------------------


class TestToHit:
    def test_maps_an_organic_result(self):
        hit = websearch._to_hit(ORGANIC, 0, "what is hnsw")
        assert hit is not None
        assert hit.source == "web"
        assert hit.url == "https://example.com/hnsw"
        assert hit.filename == "HNSW explained"
        assert hit.text == ORGANIC["snippet"]

    def test_the_same_url_always_gets_the_same_id(self):
        """The property dedupe depends on.

        `react` keys `seen` on chunk_id. With a random id per call, one page
        found by two different queries would occupy two citation slots and be
        handed to the model twice.
        """
        a = websearch._to_hit(ORGANIC, 0, "query one")
        b = websearch._to_hit({**ORGANIC, "position": 7}, 5, "a different query")
        assert a.chunk_id == b.chunk_id

    def test_different_urls_get_different_ids(self):
        other = websearch._to_hit({**ORGANIC, "link": "https://example.com/other"}, 0, "q")
        assert other.chunk_id != websearch._to_hit(ORGANIC, 0, "q").chunk_id

    @pytest.mark.parametrize(
        "bad",
        [
            {"title": "t", "link": "https://x.test", "snippet": ""},
            {"title": "t", "link": "https://x.test"},
            {"title": "t", "snippet": "text but nowhere to point"},
        ],
    )
    def test_rejects_results_that_cannot_be_cited(self, bad):
        """No snippet = nothing to ground a claim in. No link = uncitable.

        An uncitable source is worse than a missing one: it can only be used
        to support a claim the reader cannot check.
        """
        assert websearch._to_hit(bad, 0, "q") is None

    def test_falls_back_to_the_url_when_there_is_no_title(self):
        hit = websearch._to_hit({**ORGANIC, "title": ""}, 0, "q")
        assert hit.filename == "https://example.com/hnsw"

    def test_score_decreases_with_rank(self):
        first = websearch._to_hit(ORGANIC, 0, "q")
        fifth = websearch._to_hit(ORGANIC, 4, "q")
        assert first.score > fifth.score


class TestEnabled:
    def test_off_without_a_key(self, monkeypatch):
        monkeypatch.setattr(websearch, "get_settings", lambda: _settings())
        assert websearch.enabled() is False

    def test_on_with_a_key(self, monkeypatch):
        monkeypatch.setattr(websearch, "get_settings", lambda: _settings(serper_api_key="k"))
        assert websearch.enabled() is True

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_unconfigured(self, monkeypatch):
        """Never load-bearing: an unconfigured web search must degrade to
        "no results", not raise and fail the turn."""
        monkeypatch.setattr(websearch, "get_settings", lambda: _settings())
        assert await websearch.search_web("anything") == []


# --------------------------------------------------------------------------
# tool declarations
# --------------------------------------------------------------------------


class TestToolSpecs:
    def _names(self, specs) -> list[str]:
        return [d["name"] for d in specs[0]["functionDeclarations"]]

    def test_web_is_omitted_when_unconfigured(self, monkeypatch):
        """A tool the model can see but cannot use is worse than no tool: it
        keeps choosing it, gets nothing back, and burns rounds.

        The other two are unconditional -- `search_documents` is the point of
        the app, and `read_around` reads already-retrieved passages, so neither
        depends on external configuration.
        """
        monkeypatch.setattr(tools.websearch, "enabled", lambda: False)
        assert tools.SEARCH_WEB not in self._names(tools.tool_specs())

    def test_web_is_declared_when_configured(self, monkeypatch):
        monkeypatch.setattr(tools.websearch, "enabled", lambda: True)
        assert tools.SEARCH_WEB in self._names(tools.tool_specs())

    def test_the_context_ladder_is_always_available(self, monkeypatch):
        """`read_around` is the rung between "the matched passage" and "the
        whole document". Without it the model's only move when a passage refers
        to something it cannot see is to search again using terms from the very
        sentence it does not understand -- which returns the same passage."""
        monkeypatch.setattr(tools.websearch, "enabled", lambda: False)
        assert tools.READ_AROUND in self._names(tools.tool_specs())

    def test_every_declaration_is_usable(self, monkeypatch):
        """A real description, and every required field actually declared.

        This used to also demand at least one required PARAMETER, on the
        reasoning that a tool with no arguments is under-specified. The
        metadata tools disproved it: `corpus_stats` takes nothing because there
        is nothing to vary -- it reports on the whole collection, and inventing
        a parameter to satisfy a test would give the model a knob to turn
        wrongly.

        What still has to hold is that a declared requirement exists: a
        `required` naming a field absent from `properties` is a schema the API
        rejects at call time, which surfaces as the model mysteriously never
        using that tool.
        """
        monkeypatch.setattr(tools.websearch, "enabled", lambda: True)
        for decl in tools.tool_specs()[0]["functionDeclarations"]:
            assert decl["description"].strip(), decl["name"]
            params = decl["parameters"]
            assert params["type"] == "object", decl["name"]
            assert "properties" in params, decl["name"]
            for field in params.get("required", []):
                assert field in params["properties"], decl["name"]

    def test_the_description_says_when_not_to_use_it(self, monkeypatch):
        """Most "the agent chose the wrong tool" problems are description
        problems. The metadata tools sit closest to the search tools in intent,
        so theirs are the ones that must draw the line explicitly."""
        monkeypatch.setattr(tools.websearch, "enabled", lambda: True)
        by_name = {
            d["name"]: d["description"]
            for d in tools.tool_specs()[0]["functionDeclarations"]
        }
        assert "search_documents" in by_name[tools.LIST_DOCUMENTS]
        assert "Not for the contents" in by_name[tools.CORPUS_STATS]

    def test_the_search_tools_take_a_query(self, monkeypatch):
        monkeypatch.setattr(tools.websearch, "enabled", lambda: True)
        for decl in tools.tool_specs()[0]["functionDeclarations"]:
            if decl["name"] in (tools.SEARCH_DOCUMENTS, tools.SEARCH_WEB):
                assert decl["parameters"]["required"] == ["query"]


@pytest.mark.asyncio
class TestRunTool:
    async def _run(self, name, args):
        return await tools.run_tool(name, args, top_k=5, document_ids=None, owner_id=None)

    async def test_documents_returns_hits_and_text(self, monkeypatch):
        async def fake_retrieve(query, **kw):
            return [_hit(1), _hit(2)]

        monkeypatch.setattr(tools, "retrieve", fake_retrieve)
        hits, text = await self._run(tools.SEARCH_DOCUMENTS, {"query": "findings"})
        assert len(hits) == 2
        assert "passage 1" in text and "passage 2" in text

    async def test_the_model_never_sees_citation_numbers(self, monkeypatch):
        """Numbering is assigned once, at drafting time, over the whole
        accumulated evidence set. Per-round numbers would mean something
        different every round and end up pointing at the wrong sources."""

        async def fake_retrieve(query, **kw):
            return [_hit(1), _hit(2)]

        monkeypatch.setattr(tools, "retrieve", fake_retrieve)
        _, text = await self._run(tools.SEARCH_DOCUMENTS, {"query": "x"})
        assert "[1]" not in text and "[2]" not in text

    async def test_an_empty_query_is_reported_not_searched(self, monkeypatch):
        called = False

        async def fake_retrieve(query, **kw):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(tools, "retrieve", fake_retrieve)
        hits, text = await self._run(tools.SEARCH_DOCUMENTS, {"query": "   "})
        assert hits == [] and "empty" in text.lower()
        assert not called, "searched on an empty query"

    async def test_a_failing_tool_comes_back_as_text(self, monkeypatch):
        """The invariant the whole loop rests on: run_tool NEVER raises.

        An exception here would end the loop; a message the model can read
        lets it rephrase, which is the point of the observation step.
        """

        async def boom(query, **kw):
            raise RuntimeError("qdrant is down")

        monkeypatch.setattr(tools, "retrieve", boom)
        hits, text = await self._run(tools.SEARCH_DOCUMENTS, {"query": "x"})
        assert hits == []
        assert "RuntimeError" in text

    async def test_an_invented_tool_is_reported(self):
        hits, text = await self._run("search_slack", {"query": "x"})
        assert hits == [] and "Unknown tool" in text

    async def test_web_is_refused_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(tools.websearch, "enabled", lambda: False)
        hits, text = await self._run(tools.SEARCH_WEB, {"query": "x"})
        assert hits == [] and "not configured" in text


# --------------------------------------------------------------------------
# the ReAct loop
# --------------------------------------------------------------------------


class ScriptedLLM:
    """A model that replays a fixed list of (function_calls, text) per round."""

    def __init__(self, script):
        self.script = list(script)
        self.calls_seen = 0
        self.contents_at_each_round: list[int] = []

    async def generate_tools(self, contents, *, tools, system=None, **kw):
        self.calls_seen += 1
        self.contents_at_each_round.append(len(contents))
        calls, text = self.script.pop(0) if self.script else ([], "nothing more to look up")
        # A real response carries the model's own tool REQUESTS, which the
        # caller must append so the next round knows what it already asked.
        return calls, text, {"role": "model", "parts": [{"text": text}]}


def _call(query: str, name: str = tools.SEARCH_DOCUMENTS) -> dict:
    return {"name": name, "args": {"query": query}}


BASE_STATE = {"question": "who are the competitors and how are they funded"}


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted model and a retriever keyed on the query."""

    def install(script, retrieve_map=None, **settings_overrides):
        llm = ScriptedLLM(script)
        monkeypatch.setattr(react_mod, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(react_mod, "get_settings", lambda: _settings(**settings_overrides))
        monkeypatch.setattr(tools, "get_settings", lambda: _settings(**settings_overrides))

        async def fake_retrieve(query, **kw):
            return (retrieve_map or {}).get(query, [])

        monkeypatch.setattr(tools, "retrieve", fake_retrieve)
        return llm

    return install


@pytest.mark.asyncio
class TestReact:
    async def test_stops_when_the_model_asks_for_no_tools(self, scripted):
        llm = scripted([([], "the documents already cover this")])
        out = await react_mod.react(dict(BASE_STATE))
        assert llm.calls_seen == 1
        assert out["evidence"] == []

    async def test_a_later_round_sees_the_earlier_results(self, scripted):
        """The thing `plan` cannot do: choose a query using what a previous
        search returned."""
        llm = scripted(
            [
                ([_call("competitors")], ""),
                ([_call("Bolt funding")], ""),
                ([], "done"),
            ],
            {"competitors": [_hit(1)], "Bolt funding": [_hit(2)]},
        )
        out = await react_mod.react(dict(BASE_STATE))
        assert llm.calls_seen == 3
        assert [h.chunk_index for h in out["evidence"]] == [1, 2]
        # Round 2 saw more transcript than round 1: the request AND its result.
        assert llm.contents_at_each_round[1] > llm.contents_at_each_round[0]

    async def test_dedupes_the_same_chunk_across_rounds(self, scripted):
        """Two phrasings finding one chunk must not spend two citation slots,
        nor hand the model the same passage twice and inflate its weight."""
        scripted(
            [
                ([_call("first phrasing")], ""),
                ([_call("second phrasing")], ""),
                ([], "done"),
            ],
            {"first phrasing": [_hit(1)], "second phrasing": [_hit(1), _hit(2)]},
        )
        out = await react_mod.react(dict(BASE_STATE))
        assert [h.chunk_index for h in out["evidence"]] == [1, 2]

    async def test_honours_the_round_cap(self, scripted):
        """A model that never stops asking must not loop forever."""
        forever = [([_call("again")], "")] * 20
        llm = scripted(forever, {"again": [_hit(1)]}, react_max_rounds=3)
        out = await react_mod.react(dict(BASE_STATE))
        assert llm.calls_seen == 3
        # Whatever was gathered still goes to drafting: capping is not failing.
        assert len(out["evidence"]) == 1
        assert any("round_cap" in r for r in out["trace"][0]["rounds"])

    async def test_truncates_an_over_eager_round(self, scripted):
        """Truncated, not rejected -- cancelling the whole round would teach
        the model nothing and waste the searches it got right."""
        greedy = [_call(f"q{i}") for i in range(9)]
        hits = {f"q{i}": [_hit(i + 1)] for i in range(9)}
        scripted([(greedy, ""), ([], "done")], hits, react_max_calls_per_round=2)
        out = await react_mod.react(dict(BASE_STATE))
        assert len(out["evidence"]) == 2

    async def test_gathers_evidence_and_writes_no_answer(self, scripted):
        """The design decision this node exists to preserve: composition stays
        in `draft`, so citation numbering, sources_used and faithfulness all
        keep working with no special case for this path."""
        scripted([([_call("competitors")], ""), ([], "found three")], {"competitors": [_hit(1)]})
        out = await react_mod.react(dict(BASE_STATE))
        assert "answer" not in out
        assert out["evidence"] and out["pending_queries"] == []

    async def test_surfaces_its_queries_as_sub_questions(self, scripted):
        """The trace panel slot the planned path fills with its decomposition,
        so a ReAct turn is just as inspectable."""
        scripted(
            [([_call("competitors"), _call("funding")], ""), ([], "done")],
            {"competitors": [_hit(1)], "funding": [_hit(2)]},
        )
        out = await react_mod.react(dict(BASE_STATE))
        assert out["sub_questions"] == ["competitors", "funding"]

    async def test_a_model_failure_keeps_what_was_gathered(self, scripted, monkeypatch):
        """An LLMError mid-loop degrades to drafting from partial evidence,
        which beats discarding a round of real retrieval."""
        llm = scripted([([_call("competitors")], ""), ([], "")], {"competitors": [_hit(1)]})

        real = llm.generate_tools
        calls = {"n": 0}

        async def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise react_mod.LLMError("429 exhausted")
            return await real(*a, **kw)

        llm.generate_tools = flaky
        out = await react_mod.react(dict(BASE_STATE))
        assert len(out["evidence"]) == 1
        assert any("error" in r for r in out["trace"][0]["rounds"])

    async def test_counts_web_evidence_separately(self, scripted):
        """Surfaced in the trace so a reader can tell at a glance whether an
        answer leaned on their own documents or on the public web."""
        scripted(
            [([_call("hnsw", tools.SEARCH_WEB)], ""), ([], "done")],
        )

        async def fake_web(query, **kw):
            return [_hit(1, source="web")]

        # Patched on the tools module's reference, which is what run_tool uses.
        import app.services.websearch as ws

        original_enabled, original_search = ws.enabled, ws.search_web
        ws.enabled, ws.search_web = (lambda: True), fake_web
        try:
            out = await react_mod.react(dict(BASE_STATE))
        finally:
            ws.enabled, ws.search_web = original_enabled, original_search

        assert out["trace"][0]["n_web"] == 1


# --------------------------------------------------------------------------
# documents as a peer source, not a boundary
# --------------------------------------------------------------------------


class TestSourceLabelling:
    """Both kinds carry a label, and the difference survives into the prompt.

    This is what the drafter routes on when it tells the reader that a figure
    came from a public page rather than from their own file. It used to label
    only `web`, leaving "document" to be inferred from the ABSENCE of a mark --
    which is not something a prompt can rely on.
    """

    def test_a_document_source_is_labelled(self):
        from app.services.retrieval import build_context

        out = build_context([_hit(1)])
        assert out.startswith("[1] document · doc1.pdf")

    def test_a_web_source_is_labelled_and_carries_its_url(self):
        from app.services.retrieval import build_context

        hit = websearch._to_hit(ORGANIC, 0, "q")
        out = build_context([hit])
        assert "[1] web · HNSW explained · https://example.com/hnsw" in out

    def test_the_two_kinds_are_distinguishable_in_one_context(self):
        from app.services.retrieval import build_context

        out = build_context([_hit(1), websearch._to_hit(ORGANIC, 0, "q")])
        assert "[1] document ·" in out and "[2] web ·" in out


class TestReactIsTheDefault:
    def test_react_is_on_by_default(self):
        """Documents are one source, not the limit. Only the ReAct path can
        reach anything else, so it has to be the default for the assistant to
        behave as described."""
        assert _settings().react_default is True

    @pytest.mark.asyncio
    async def test_the_evaluation_harness_pins_the_planned_path(self, monkeypatch):
        """The guard for the trap this flip walked into.

        "agent" in the harness means the plan/retrieve/draft/critique graph --
        that is what every recorded recall and faithfulness number measures.
        Leaving `react` unset here would let a config default redefine the
        thing under test, so old and new runs would compare different systems
        from the same column.
        """
        import app.agent.graph as graph_mod
        from app.services import tier2

        seen = {}

        async def fake_run_agent(question, **kw):
            seen.update(kw)

            class R:
                answer, evidence, iterations, sufficient = "a", [], 0, True
                sub_questions, citations, critique = [], [], ""

            return R()

        monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
        await tier2._generate(
            tier2.GoldenQuestion(id="q1", question="anything"),
            mode="agent",
            top_k=5,
            multi_query=False,
            owner_id=None,
        )
        assert seen.get("react") is False, "the harness inherited REACT_DEFAULT"


class TestCitationsDerivedFromText:
    """The inline [n] markers are the fallback when `sources_used` is lost.

    That list is emitted LAST in the JSON, so a truncated response drops it
    while keeping the prose -- which produced answers citing [1] and [2] in
    every sentence and reporting "no sources cited". The markers are what the
    reader sees and what the UI renders, so they are the better ground truth.
    """

    def test_reads_plain_markers(self):
        from app.agent.nodes import _cited_in_text

        assert _cited_in_text("Pool exhaustion [1]. Retries lacked jitter [2].", 5) == [1, 2]

    def test_reads_grouped_markers(self):
        from app.agent.nodes import _cited_in_text

        assert _cited_in_text("Two pools were opened [1, 2].", 5) == [1, 2]

    def test_dedupes_and_preserves_order(self):
        from app.agent.nodes import _cited_in_text

        assert _cited_in_text("a [3] b [1] c [3] d [1]", 5) == [3, 1]

    def test_drops_out_of_range_numbers(self):
        """A [7] against six sources is a hallucinated citation. Passing it on
        would have the UI resolve it to nothing, or to the wrong source."""
        from app.agent.nodes import _cited_in_text

        assert _cited_in_text("real [2] invented [9] zero [0]", 6) == [2]

    def test_no_markers_is_empty(self):
        """So a genuinely uncited answer still reports as uncited -- the badge
        and the critique retry both depend on that staying true."""
        from app.agent.nodes import _cited_in_text

        assert _cited_in_text("An answer with nothing to back it.", 5) == []


class CapturingLLM:
    """Records the prompt it was given, then returns a fixed JSON reply."""

    def __init__(self, reply='{"ambiguous": false}'):
        self.reply = reply
        self.prompt = ""
        self.system = ""

    async def generate(self, prompt, *, schema=None, system=None, **kw):
        self.prompt = prompt
        self.system = system or ""
        return self.reply


@pytest.mark.asyncio
class TestDraftKnowsItsCoverage:
    """The drafter is told what it ACTUALLY did, not what it could have done.

    THE BUG THESE PIN

    The coverage line used to read "Both the user's documents and the web were
    SEARCHABLE for this question" whenever web search was configured -- a
    statement about capability. The answer rules tell the model to report its
    work from that line, so a turn that touched only the metadata tools opened
    with "I searched your documents and the web." Nothing had been searched.

    Asked about it on the next turn, the model correctly said no web search had
    happened and called its own opening a "boilerplate artifact". That reads as
    a model contradicting itself; it was in fact being told two different
    things and reporting each of them faithfully.

    Availability is not use. These assert that distinction in both directions.
    """

    async def _prompt_for(self, monkeypatch, *, web_enabled, evidence=None, facts=None):
        from app.agent import nodes

        llm = CapturingLLM('{"answer": "x [1]", "sources_used": [1]}')
        monkeypatch.setattr(nodes, "get_pool", lambda role: _StubPool(llm))
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: web_enabled)
        state = {
            "question": "q",
            "evidence": [_hit(1)] if evidence is None else evidence,
            "chat_context": "",
        }
        if facts:
            state["corpus_facts"] = facts
        await nodes.draft(state)
        return llm.prompt

    async def test_says_so_when_web_is_unavailable(self, monkeypatch):
        prompt = await self._prompt_for(monkeypatch, web_enabled=False)
        assert "NOT configured" in prompt
        assert "do not imply the information does not exist" in prompt

    async def test_an_available_web_is_not_reported_as_a_search(self, monkeypatch):
        """The exact regression. Available and unused must not read as used."""
        prompt = await self._prompt_for(monkeypatch, web_enabled=True)
        assert "searched the web" not in prompt
        assert "was NOT used" in prompt

    async def test_a_web_hit_is_reported_as_a_web_search(self, monkeypatch):
        prompt = await self._prompt_for(
            monkeypatch,
            web_enabled=True,
            evidence=[_hit(1), _hit(2, source="web")],
        )
        assert "searched the web" in prompt
        assert "searched the user's documents" in prompt

    async def test_metadata_only_is_not_reported_as_searching_anything(
        self, monkeypatch
    ):
        """Counting documents is not reading them.

        This is the turn that produced the false claim: "how many docs do we
        have" resolved entirely from the metadata tools, with no retrieval at
        all, and was reported as a search of both the corpus and the web.
        """
        prompt = await self._prompt_for(
            monkeypatch,
            web_enabled=True,
            evidence=[],
            facts=["10 documents in the collection."],
        )
        assert "WITHOUT searching their contents" in prompt
        assert "searched the web" not in prompt

    async def test_resolve_is_given_the_same_line(self, monkeypatch):
        """`resolve` shares ANSWER_RULES, which cite the coverage line.

        It was never given one, so every turn that exhausted the critique loop
        invented its opening sentence out of nothing. A rule referring to a
        block that only one of its two readers receives is the same drift that
        put the formatting rules in `draft` alone.
        """
        from app.agent import nodes

        llm = CapturingLLM('{"answer": "x [1]", "sources_used": [1]}')
        monkeypatch.setattr(nodes, "get_pool", lambda role: _StubPool(llm))
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: True)
        await nodes.resolve(
            {
                "question": "q",
                "evidence": [_hit(1)],
                "draft": "d",
                "missing": ["m"],
                "chat_context": "",
            }
        )
        assert "Search coverage" in llm.prompt


@pytest.mark.asyncio
class TestClarifySeesTheConversation:
    async def test_the_conversation_reaches_the_clarifier(self, monkeypatch):
        """The omission that produced the app's worst clarifying question.

        Asked "then answer from the internet!" as a follow-up about a pyramid,
        `clarify` saw six words and no topic and invented three categories:
        "Latest Technology Trends", "Global Financial Markets", "Historical
        Architecture". Every other node already got history; the one node whose
        job is judging whether a request is clear was judging it in isolation.
        """
        from app.agent import nodes

        llm = CapturingLLM()
        monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: False)

        async def no_outline(*a, **kw):
            return []

        monkeypatch.setattr(nodes, "document_outline", no_outline)

        await nodes.clarify(
            {
                "question": "then answer from the internet!",
                "clarify": True,
                "chat_context": "user: Tell me about Akhet Khufu",
            }
        )
        assert "Akhet Khufu" in llm.prompt, "the clarifier judged it in isolation"

    async def test_does_not_offer_the_web_when_it_is_unreachable(self, monkeypatch):
        """An option requiring public information is worthless on a deployment
        that cannot reach it, and the clarifier has no other way to know."""
        from app.agent import nodes

        llm = CapturingLLM()
        monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(nodes.websearch, "enabled", lambda: False)

        async def no_outline(*a, **kw):
            return []

        monkeypatch.setattr(nodes, "document_outline", no_outline)

        await nodes.clarify({"question": "q", "clarify": True, "chat_context": ""})
        assert "web search is not configured" in llm.prompt


class _StubPool:
    """Stands in for ModelPool so a test can stub the model behind it.

    The nodes now go through a pool rather than calling get_llm directly -- the
    pool spreads load across models sharing one role. Tests stub the pool at the
    same seam so they exercise the production path rather than a bypass.
    """

    def __init__(self, client):
        self._client = client

    def for_prompt(self, *a, **kw):
        return self._client

    def pick(self, tokens):
        return self._client

    async def generate(self, prompt, **kwargs):
        # The pool now owns the call so it can hand off to another model on a
        # 429. Delegating keeps the stub a seam rather than a second
        # implementation.
        kwargs.pop("max_attempts", None)
        return await self._client.generate(prompt, **kwargs)
