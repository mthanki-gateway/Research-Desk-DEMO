"""Howler's designer, and the two rules a link depends on.

Both are cheap to break by being helpful, and neither shows up as an error --
they show up as an interview that asks the wrong things, weeks later.
"""

import app.api.howler as howler
from app.services import designer


class TestDesignerSchema:
    """What the model is allowed to hand back, and why each key is shaped so."""

    def test_fields_are_required(self):
        """Or the cheapest valid completion is a reply and nothing else.

        With only `reply` required that is exactly what came back, turn after
        turn -- including turns whose reply claimed to have drafted data points
        while returning none.
        """
        assert set(designer.SCHEMA["required"]) == {"reply", "fields"}

    def test_there_is_no_title(self):
        """Naming is what sent this model into repetition loops.

        Asked for a title inline it produced sixty words of near-synonyms one
        turn and a Korean phrase thirty times the next, which ate the token
        budget and truncated the JSON. `title_for` does it in its own call.
        """
        assert "title" not in designer.SCHEMA["properties"]

    def test_the_keys_that_change_behaviour_are_described(self):
        """An undescribed key is one the model never touches.

        `research` and `synthesise` are the two that change what HAPPENS rather
        than what gets stored. Left bare, `research` went unused no matter how
        firmly the system prompt asked for it.
        """
        for key in ("research", "synthesise"):
            assert designer.SCHEMA["properties"][key].get("description"), key

    def test_it_knows_where_the_links_and_results_are(self):
        """So it can say where to click, rather than describing a schema.

        It is the Design tab of a page with two others; a designer that hands
        over a link without saying where to find it has hidden its own output.
        """
        assert "WHERE THINGS ARE ON THEIR SCREEN" in designer.SYSTEM
        assert "under Links" in designer.SYSTEM


class TestAutoSynthesise:
    def test_it_cannot_finish_with_nothing_to_finish(self):
        """`synthesise` only counts alongside a schema.

        A model asking to settle while returning no fields is asking for an
        empty interview, and the link would gather nothing.
        """
        assert designer.SCHEMA["properties"]["synthesise"]["type"] == "boolean"


class TestTheAutoLinkCarriesNoParticipant:
    """The subtle one, and the reason this file exists.

    An invite's own `participant` OVERRIDES the project's when a guest
    connects. That is right for a link made for a named person, and wrong for
    the one the designer makes itself -- which is minted the moment the schema
    settles, often on the first turn, from a paragraph that is still a guess.
    Copying it across freezes that guess and no later refinement ever reaches
    the interviewer.
    """

    def test_live_invite_takes_only_a_project(self):
        """Signature enforced, because the bug was an extra argument.

        `_live_invite(project_id, row.participant)` looked obviously correct
        and quietly pinned every auto-made link to a first draft.
        """
        import inspect

        params = list(inspect.signature(howler._live_invite).parameters)
        assert params == ["project_id"], params

    def test_the_precedence_it_relies_on_is_documented(self):
        assert "OVERRIDES the project's" in howler._live_invite.__doc__
