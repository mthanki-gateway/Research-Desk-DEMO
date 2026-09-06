"""Human-in-the-loop: asking the user what they meant, then continuing.

Two layers, deliberately separated.

`apply_clarification` and `_clean_options` are PURE, so every decision shape can
be checked without a graph, a checkpointer or an LLM. That is why the
interesting logic lives there rather than inside the nodes.

The graph tests then prove what the pure functions cannot: that `interrupt()`
really stops execution, that state survives, that `Command(resume=...)`
continues from the checkpoint, and -- the one that motivated splitting `clarify`
from `ask_human` -- that the EXPENSIVE node does not run twice.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agent.graph import after_human, interrupt_payload, needs_human
from app.agent.nodes import _clean_options, apply_clarification, ask_human
from app.agent.state import ResearchState

ORIGINAL = "Tell me about the pyramids, the specific thing i want to know"
OPTIONS = [
    {"label": "Construction methods", "description": "How the blocks were moved"},
    {"label": "Who built them", "description": "Workforce and organisation"},
]


# --------------------------------------------------------------------------
# option cleaning
# --------------------------------------------------------------------------


class TestCleanOptions:
    def test_keeps_well_formed_pairs(self):
        assert _clean_options(OPTIONS) == OPTIONS

    def test_drops_options_with_no_label(self):
        # A label is what the user clicks. Without one there is nothing to show.
        assert _clean_options([{"description": "x"}, {"label": "  ", "description": "y"}]) == []

    def test_supplies_a_missing_description(self):
        assert _clean_options([{"label": "Costs"}]) == [
            {"label": "Costs", "description": ""}
        ]

    def test_caps_at_four(self):
        """More than four choices is a menu, not a question."""
        many = [{"label": f"o{i}", "description": ""} for i in range(9)]
        assert len(_clean_options(many)) == 4

    @pytest.mark.parametrize("raw", [None, "options", 7, {"label": "x"}])
    def test_non_list_input_is_empty(self, raw):
        assert _clean_options(raw) == []

    def test_skips_non_dict_entries(self):
        assert _clean_options(["a string", {"label": "Costs", "description": "d"}]) == [
            {"label": "Costs", "description": "d"}
        ]


# --------------------------------------------------------------------------
# the pure decision logic
# --------------------------------------------------------------------------


class TestApplyClarification:
    def test_answer_rewrites_the_question(self):
        update = apply_clarification(
            {"action": "answer", "answer": "Construction methods"}, ORIGINAL
        )
        # The rewritten question is the entire point: plan, retrieval and
        # drafting all read `question`, so narrowing it here narrows everything
        # downstream without any other node knowing clarification exists.
        assert update["question"].startswith(ORIGINAL)
        assert "Construction methods" in update["question"]
        assert update["clarification"] == "Construction methods"
        assert update["original_question"] == ORIGINAL

    def test_custom_text_is_not_a_special_case(self):
        """Free text and a chosen label take the identical path."""
        typed = apply_clarification(
            {"action": "answer", "answer": "the casing stones"}, ORIGINAL
        )
        chosen = apply_clarification(
            {"action": "answer", "answer": "Who built them"}, ORIGINAL
        )
        assert set(typed) == set(chosen)

    def test_skip_leaves_the_question_untouched(self):
        update = apply_clarification({"action": "skip"}, ORIGINAL)
        assert "question" not in update
        assert "clarification" not in update

    def test_cancel_stops_without_searching(self):
        update = apply_clarification({"action": "cancel"}, ORIGINAL)
        assert update["cancelled"] is True
        assert update["pending_queries"] == []
        # `draft` is what the API reads as the answer, and every node that
        # would write it is about to be skipped.
        assert update["draft"]
        assert update["sufficient"] is True

    def test_answer_with_empty_text_is_a_skip(self):
        """Submitting a blank box must not narrow the search to nothing."""
        update = apply_clarification({"action": "answer", "answer": "   "}, ORIGINAL)
        assert "question" not in update

    @pytest.mark.parametrize(
        "decision",
        [None, "answer", 42, {}, {"action": "nonsense"}, {"answer": "x"}],
    )
    def test_malformed_decisions_degrade_to_skip(self, decision):
        """Never fail a turn over a bad payload.

        Skip is what the graph would have done with clarification switched off,
        so a malformed decision lands on the pre-existing behaviour rather than
        on a new one.
        """
        update = apply_clarification(decision, ORIGINAL)
        assert "question" not in update
        assert not update.get("cancelled")

    def test_action_is_case_insensitive(self):
        assert apply_clarification({"action": " Cancel "}, ORIGINAL)["cancelled"] is True


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


class TestRouting:
    def test_asks_only_when_a_question_was_drafted(self):
        assert needs_human({"pending_clarification": {"question": "?"}}) == "ask_human"

    def test_clear_question_goes_straight_to_plan(self):
        assert needs_human({}) == "plan"

    def test_empty_clarification_goes_straight_to_plan(self):
        """`clarify` writes nothing when the request is already specific."""
        assert needs_human({"pending_clarification": {}}) == "plan"

    def test_cancel_ends_the_run(self):
        assert after_human({"cancelled": True}) == END

    def test_otherwise_plans(self):
        assert after_human({"cancelled": False}) == "plan"
        assert after_human({}) == "plan"


# --------------------------------------------------------------------------
# interrupt payload extraction
# --------------------------------------------------------------------------


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class TestInterruptPayload:
    def test_none_when_not_paused(self):
        assert interrupt_payload({"draft": "done"}) is None

    def test_empty_tuple_is_not_paused(self):
        assert interrupt_payload({"__interrupt__": ()}) is None

    def test_extracts_value_from_a_tuple(self):
        payload = {"type": "clarification", "options": OPTIONS}
        assert interrupt_payload({"__interrupt__": (_FakeInterrupt(payload),)}) == payload

    def test_accepts_a_bare_object(self):
        payload = {"type": "clarification"}
        assert interrupt_payload({"__interrupt__": _FakeInterrupt(payload)}) == payload

    def test_non_dict_value_is_wrapped_rather_than_raising(self):
        """A shape change upstream must not raise inside an API handler."""
        out = interrupt_payload({"__interrupt__": (_FakeInterrupt("a string"),)})
        assert out["type"] == "unknown"


# --------------------------------------------------------------------------
# the graph actually pausing and resuming
# --------------------------------------------------------------------------

# Counts executions of the expensive node, which is what proves the two-node
# split works. A module-level list rather than a fixture so the stub closure
# stays trivial.
CALLS: list[str] = []


def _stub_graph():
    """clarify -> ask_human -> plan, with only `ask_human` being real.

    `clarify` and `plan` are stand-ins: what is under test is the pause/resume
    mechanism and the node boundary, not what any model produces.
    """

    async def fake_clarify(state):
        CALLS.append("clarify")
        if not state.get("clarify"):
            return {}
        return {"pending_clarification": {"question": "Which aspect?", "options": OPTIONS}}

    async def fake_plan(state):
        CALLS.append("plan")
        # Echoes the question retrieval would actually run, so a test can see
        # whether the clarification reached it.
        return {"draft": state["question"]}

    builder = StateGraph(ResearchState)
    builder.add_node("clarify", fake_clarify)
    builder.add_node("ask_human", ask_human)
    builder.add_node("plan", fake_plan)
    builder.add_edge(START, "clarify")
    builder.add_conditional_edges(
        "clarify", needs_human, {"ask_human": "ask_human", "plan": "plan"}
    )
    builder.add_conditional_edges("ask_human", after_human, {"plan": "plan", END: END})
    builder.add_edge("plan", END)
    return builder.compile(checkpointer=MemorySaver())


CONFIG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.mark.asyncio
class TestGraphPauseResume:
    async def test_runs_straight_through_when_clarify_is_off(self):
        graph = _stub_graph()
        out = await graph.ainvoke({"question": ORIGINAL, "clarify": False}, config=CONFIG)
        assert interrupt_payload(out) is None
        assert out["draft"] == ORIGINAL

    async def test_pauses_before_planning_when_the_question_is_vague(self):
        graph = _stub_graph()
        out = await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)

        payload = interrupt_payload(out)
        assert payload is not None
        assert payload["type"] == "clarification"
        assert payload["question"] == "Which aspect?"
        assert payload["options"] == OPTIONS
        assert payload["original"] == ORIGINAL
        # The decisive assertion: planning has NOT run. Asking is only worth
        # anything if it happens before the work it would redirect.
        assert "draft" not in out
        assert "plan" not in CALLS

    async def test_answer_reaches_the_planner(self):
        graph = _stub_graph()
        await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)

        out = await graph.ainvoke(
            Command(resume={"action": "answer", "answer": "Construction methods"}),
            config=CONFIG,
        )
        # fake_plan echoes the question it was given.
        assert "Construction methods" in out["draft"]
        assert out["clarification"] == "Construction methods"

    async def test_skip_plans_the_original_question(self):
        graph = _stub_graph()
        await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)

        out = await graph.ainvoke(Command(resume={"action": "skip"}), config=CONFIG)
        assert out["draft"] == ORIGINAL
        assert "clarification" not in out

    async def test_cancel_never_plans(self):
        graph = _stub_graph()
        await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)

        out = await graph.ainvoke(Command(resume={"action": "cancel"}), config=CONFIG)
        assert out["cancelled"] is True
        assert "plan" not in CALLS

    async def test_the_expensive_node_runs_exactly_once(self):
        """The reason `clarify` and `ask_human` are separate nodes.

        `interrupt()` does not suspend a function mid-body: on resume the node
        re-executes FROM THE TOP. Had the LLM call that drafts the options
        lived in the same node as the interrupt, answering would pay for it a
        second time -- silently, and only on the human-in-the-loop path.
        """
        graph = _stub_graph()
        await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)
        assert CALLS.count("clarify") == 1

        await graph.ainvoke(Command(resume={"action": "skip"}), config=CONFIG)
        assert CALLS.count("clarify") == 1, "clarify re-ran on resume"

    async def test_state_survives_between_the_two_calls(self):
        """The pause is durable, not a parked coroutine.

        The resume below is a completely separate `ainvoke`. It works only
        because the checkpointer persisted the state -- the same property that
        lets a real resume arrive from a different worker, after a deploy.
        """
        graph = _stub_graph()
        await graph.ainvoke({"question": ORIGINAL, "clarify": True}, config=CONFIG)

        snapshot = await graph.aget_state(CONFIG)
        assert snapshot.next == ("ask_human",)
        assert snapshot.values["question"] == ORIGINAL

        out = await graph.ainvoke(Command(resume={"action": "skip"}), config=CONFIG)
        assert out["question"] == ORIGINAL

    async def test_threads_are_independent(self):
        """One paused turn must not be resumed by another's answer."""
        graph = _stub_graph()
        a = {"configurable": {"thread_id": "a"}}
        b = {"configurable": {"thread_id": "b"}}

        await graph.ainvoke({"question": "qa", "clarify": True}, config=a)
        await graph.ainvoke({"question": "qb", "clarify": True}, config=b)

        out_a = await graph.ainvoke(Command(resume={"action": "cancel"}), config=a)
        assert out_a["cancelled"] is True

        snapshot_b = await graph.aget_state(b)
        assert snapshot_b.next == ("ask_human",)


# --------------------------------------------------------------------------
# option sanitising
# --------------------------------------------------------------------------


class TestOptionSanitising:
    """Gemma leaks LaTeX into string fields; a garbled option must not reach
    the user. The literal below was rendered in the UI."""

    GARBLED = "$$ ext{Norwegian Cod Fisheries and Ancient Monuments}}{ ext{"

    def test_rejects_the_real_garbled_label(self):
        assert _clean_options([{"label": self.GARBLED, "description": "x"}]) == []

    def test_keeps_the_good_options_beside_a_garbled_one(self):
        """Rejecting one option must not discard the rest of the list."""
        out = _clean_options(
            [
                {"label": "Payments Cutover Outage", "description": "The incident"},
                {"label": self.GARBLED, "description": "x"},
                {"label": "Annual Report", "description": "FY2024"},
            ]
        )
        assert [o["label"] for o in out] == [
            "Payments Cutover Outage",
            "Annual Report",
        ]

    def test_unwraps_a_recoverable_tex_label(self):
        assert _clean_options([{"label": r"\text{Financial Summary}"}]) == [
            {"label": "Financial Summary", "description": ""}
        ]

    def test_rejects_an_over_long_label(self):
        """A label is 2-6 words. A paragraph in that slot is a schema miss."""
        assert _clean_options([{"label": "word " * 40}]) == []

    def test_drops_a_garbled_description_but_keeps_the_label(self):
        out = _clean_options([{"label": "Annual Report", "description": r"\text{x}}{"}])
        assert out == [{"label": "Annual Report", "description": ""}]

    def test_dedupes_labels_case_insensitively(self):
        out = _clean_options(
            [{"label": "Annual Report"}, {"label": "annual report"}]
        )
        assert len(out) == 1
