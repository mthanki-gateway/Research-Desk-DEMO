"""Phase 2: budgets, failure-mode routing, and never a hard stop.

The router is pure and the interesting decisions all live in it, so these need
no model and no graph. What they pin is that the REMEDY MATCHES THE FAILURE --
the thing a single `sufficient: false` could not express, and the reason
`critique` now returns a category.
"""

import uuid

import pytest

from app.agent.graph import after_react, entry, should_continue
from app.agent.nodes import MISSING_EVIDENCE, UNANSWERABLE, UNSUPPORTED_CLAIM, resolve
from app.agent.state import _union, merge_evidence
from app.config import Settings
from app.services.retrieval import exclude_seen
from app.services.vectorstore import SearchHit

END = "__end__"


def _hit(n: int) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(int=n),
        document_id=uuid.UUID(int=99),
        filename=f"doc{n}.md",
        page=1,
        chunk_index=n,
        heading="## Findings",
        text=f"passage {n}",
        score=0.7,
        meta={},
        found_by=["q"],
    )


@pytest.fixture
def settings(monkeypatch):
    import app.agent.graph as graph_mod

    s = Settings(google_api_key="x")
    monkeypatch.setattr(graph_mod, "get_settings", lambda: s)
    return s


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


class TestRemedyMatchesFailure:
    def test_a_good_answer_ends(self, settings):
        assert should_continue({"sufficient": True}) == END

    def test_an_overclaim_regenerates_rather_than_searching(self, settings):
        """Searching again cannot fix a sentence that says more than its
        source -- the passages were already right. Before failure modes existed
        this went to `retrieve`, which was the wrong remedy every time."""
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": UNSUPPORTED_CLAIM,
                    "regen_count": 0,
                    "pending_queries": ["something"],
                }
            )
            == "draft"
        )

    def test_a_gap_searches_rather_than_rewording(self, settings):
        """Rewriting cannot fix words that were never in any passage."""
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": MISSING_EVIDENCE,
                    "iterations": 0,
                    "pending_queries": ["capex guidance 2025"],
                }
            )
            == "retrieve"
        )

    def test_unanswerable_goes_straight_to_resolve(self, settings):
        assert (
            should_continue({"sufficient": False, "failure_mode": UNANSWERABLE})
            == "resolve"
        )

    def test_an_unknown_mode_searches(self, settings):
        """The conservative default: looking again is recoverable, declaring a
        question unanswerable is not."""
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": "something-new",
                    "iterations": 0,
                    "pending_queries": ["q"],
                }
            )
            == "retrieve"
        )


class TestBudgets:
    def test_regens_are_capped(self, settings):
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": UNSUPPORTED_CLAIM,
                    "regen_count": settings.agent_max_regens,
                }
            )
            == "resolve"
        )

    def test_iterations_are_capped(self, settings):
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": MISSING_EVIDENCE,
                    "iterations": settings.agent_max_iterations,
                    "pending_queries": ["q"],
                }
            )
            == "resolve"
        )

    def test_regen_and_iteration_budgets_are_independent(self, settings):
        """A rewrite costs one call and no retrieval. Charging it to the
        retrieval cycle would let a single over-claim consume the turn's whole
        ability to search."""
        state = {
            "sufficient": False,
            "failure_mode": MISSING_EVIDENCE,
            "iterations": 0,
            "regen_count": settings.agent_max_regens,
            "pending_queries": ["q"],
        }
        assert should_continue(state) == "retrieve"

    def test_nothing_left_to_search_resolves(self, settings):
        assert (
            should_continue(
                {
                    "sufficient": False,
                    "failure_mode": MISSING_EVIDENCE,
                    "iterations": 0,
                    "pending_queries": [],
                }
            )
            == "resolve"
        )

    def test_an_exhausted_budget_never_ends_silently(self, settings):
        """Every exhausted path lands on `resolve`, not END.

        Ending would ship whichever draft happened to exist -- including the
        claims the critic had just rejected. A budget running out is not a
        reason to publish a criticised answer unchanged.
        """
        for mode in (UNSUPPORTED_CLAIM, MISSING_EVIDENCE, UNANSWERABLE):
            route = should_continue(
                {
                    "sufficient": False,
                    "failure_mode": mode,
                    "iterations": 99,
                    "regen_count": 99,
                    "pending_queries": [],
                }
            )
            assert route == "resolve", mode


# --------------------------------------------------------------------------
# never a hard stop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolve:
    async def test_no_evidence_abstains_and_says_what_was_searched(self):
        """The ONE hard stop. With no passages there is nothing to be partially
        right about -- but it still reports what was tried, so the user has a
        next move rather than a dead end."""
        out = await resolve(
            {
                "question": "what is the CEO's compensation?",
                "evidence": [],
                "tried_queries": ["CEO compensation", "executive pay"],
            }
        )
        assert out["partial"] is False
        assert out["citations"] == []
        assert "CEO compensation" in out["draft"]
        assert out["sufficient"] is True, "abstention is final, not a retry"

    async def test_partial_answers_keep_what_was_supported(self, monkeypatch):
        import app.agent.nodes as nodes

        class FakeLLM:
            async def generate(self, prompt, **kw):
                assert "unsupported" in prompt.lower()
                return (
                    '{"answer": "Margin rose to 62.1% [1]. Capex is not '
                    'stated.", "sources_used": [1]}'
                )

        monkeypatch.setattr(nodes, "get_pool", lambda role: _StubPool(FakeLLM()))
        out = await resolve(
            {
                "question": "margin and capex?",
                "evidence": [_hit(1)],
                "draft": "Margin rose to 62.1% [1] and capex will fall.",
                "missing": ["capex guidance"],
                "unsupported_claims": ["capex will fall"],
            }
        )
        assert out["partial"] is True
        assert "62.1%" in out["draft"]
        assert out["citations"] == [1]

    async def test_a_failed_rewrite_falls_back_to_the_draft(self, monkeypatch):
        """An imperfect grounded answer beats an error message -- which is why
        this keeps the draft rather than surfacing the failure."""
        import app.agent.nodes as nodes

        class BoomLLM:
            async def generate(self, prompt, **kw):
                raise nodes.LLMError("429")

        monkeypatch.setattr(nodes, "get_pool", lambda role: _StubPool(BoomLLM()))
        out = await resolve(
            {
                "question": "q",
                "evidence": [_hit(1)],
                "draft": "The original draft [1].",
                "citations": [1],
            }
        )
        assert out["draft"] == "The original draft [1]."


# --------------------------------------------------------------------------
# not re-serving what was already shown
# --------------------------------------------------------------------------


class TestSeenChunkIds:
    def test_excluded_from_a_retry(self):
        hits = [_hit(1), _hit(2), _hit(3)]
        kept = exclude_seen(hits, {str(uuid.UUID(int=2))})
        assert [h.chunk_index for h in kept] == [1, 3]

    def test_empty_set_is_a_no_op(self):
        hits = [_hit(1), _hit(2)]
        assert exclude_seen(hits, set()) is hits
        assert exclude_seen(hits, None) is hits

    def test_the_reducer_accumulates_across_hops(self):
        assert _union({"a"}, {"b"}) == {"a", "b"}

    def test_the_reducer_is_commutative(self):
        """LangGraph merges branches in no guaranteed order, so a reducer that
        cared about order would give different results run to run."""
        assert _union({"a"}, {"b"}) == _union({"b"}, {"a"})

    def test_without_exclusion_a_retry_dedupes_back_to_nothing(self):
        """Why this exists at all.

        `merge_evidence` dedupes by chunk_id, so a retry that re-finds the same
        passages contributes zero new evidence -- the cycle costs a full round
        of model calls and changes nothing. Exclusion is what makes a second
        pass able to differ.
        """
        first = [_hit(1), _hit(2)]
        assert merge_evidence(first, [_hit(1), _hit(2)]) == first


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


class TestAgentDecidesWhetherToRetrieve:
    """The agent's own judgement is what routes the turn, not a node in front
    of it. These pin the two edges that judgement feeds.
    """

    def test_a_direct_answer_ends_the_turn(self):
        """"hi" used to reach `draft` with no evidence, which either said
        nothing was found or answered from whatever was lexically nearest --
        three sub-questions and five web searches for a greeting."""
        assert after_react({"answered_directly": True, "evidence": []}) == END

    def test_an_empty_search_still_drafts(self):
        """THE DISTINCTION THAT MATTERS. "No evidence" has two causes: the
        agent never looked, or it looked and found nothing. Only the first is
        safe to answer in the model's own words -- the second must reach
        `draft`, where "your documents cover X but not Z" is written and where
        `critique` reviews the result. Conflating them turns a failed search
        into licence to answer from memory."""
        assert after_react({"evidence": [], "sufficient": True}) == "draft"

    def test_evidence_always_drafts(self):
        assert after_react({"evidence": [_hit(1)]}) == "draft"

    def test_the_agent_owns_routing_when_it_has_tools(self, monkeypatch):
        """No standalone router call on a turn the agent can handle itself."""
        monkeypatch.setattr(
            "app.agent.graph.get_settings",
            lambda: Settings(model_profile="gemini"),
        )
        assert entry({"react": True}) == "clarify"

    def test_the_router_stays_for_models_without_tool_calling(self, monkeypatch):
        """Gemma emits no functionCall parts, so `remember_preference` is never
        called there. Skipping the router too would leave that profile with
        nothing able to capture an instruction."""
        monkeypatch.setattr(
            "app.agent.graph.get_settings", lambda: Settings(model_profile="gemma")
        )
        assert entry({"react": True}) == "route"

    def test_the_router_stays_on_the_planned_path(self, monkeypatch):
        """The evaluation harness runs `plan`, which has no tools at all."""
        monkeypatch.setattr(
            "app.agent.graph.get_settings",
            lambda: Settings(model_profile="gemini"),
        )
        assert entry({"react": False}) == "route"
