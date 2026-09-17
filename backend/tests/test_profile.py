"""The interview's structured output, and the judgement calls inside it.

The profile is built by a TOOL the model calls as it learns things, rather than
parsed out of the transcript afterwards. Two consequences are worth pinning:
the merge has to survive how people actually talk, and the tool's RESULT is
what steers the next question.
"""

import pytest

from app.services import profile


class TestMerge:
    """People correct themselves, repeat themselves, and answer in pieces."""

    def test_a_correction_wins(self):
        """"About five years. Sorry, six."

        The last answer is the true one for a scalar. Keeping the first would
        record a number the person explicitly retracted.
        """
        p = profile.merge({"years_experience": 5}, {"years_experience": 6})
        assert p["years_experience"] == 6

    def test_lists_accumulate_rather_than_replace(self):
        """A second mention of skills is nearly always ADDITIONAL.

        Replacing would silently drop everything said the first time, and the
        loss is invisible -- the profile still looks plausible.
        """
        p = profile.merge({}, {"skills": ["Python", "Postgres"]})
        p = profile.merge(p, {"skills": ["Kubernetes"]})
        assert p["skills"] == ["Python", "Postgres", "Kubernetes"]

    def test_a_repeated_skill_is_not_duplicated(self):
        p = profile.merge({"skills": ["Python"]}, {"skills": ["Python", "Go"]})
        assert p["skills"] == ["Python", "Go"]

    def test_case_does_not_create_a_duplicate(self):
        """"Python" in one turn and "python" in the next is one skill."""
        p = profile.merge({"skills": ["Python"]}, {"skills": ["python"]})
        assert p["skills"] == ["Python"]

    def test_empty_values_do_not_erase(self):
        """A tool call that omits a field must not blank it.

        The model calls this repeatedly with whatever it just learned, so every
        call omits most fields. Treating absence as deletion would leave only
        the most recent answer.
        """
        p = profile.merge({"full_name": "Sam"}, {"full_name": "", "skills": []})
        assert p["full_name"] == "Sam"
        assert "skills" not in p

    def test_none_does_not_erase(self):
        p = profile.merge({"work_setup": "hybrid"}, {"work_setup": None})
        assert p["work_setup"] == "hybrid"

    def test_the_original_is_not_mutated(self):
        """The caller holds the row's value; merging must not edit it in place."""
        original = {"skills": ["Python"]}
        profile.merge(original, {"skills": ["Go"]})
        assert original == {"skills": ["Python"]}


class TestCompleteness:
    def test_an_empty_profile_is_missing_everything_required(self):
        assert set(profile.missing({})) == set(profile.REQUIRED)

    def test_optional_fields_never_block(self):
        """An interviewer that demands every field interrogates."""
        full = {name: "x" for name in profile.REQUIRED}
        assert profile.complete(full)
        assert "location" not in profile.missing(full)

    def test_zero_years_counts_as_answered(self):
        """A graduate with no professional experience HAS answered.

        A plain falsy check treats 0 as missing and asks again for ever, which
        is both wrong and insulting.
        """
        p = {name: "x" for name in profile.REQUIRED}
        p["years_experience"] = 0
        assert "years_experience" not in profile.missing(p)
        assert profile.complete(p)

    def test_an_empty_list_is_not_an_answer(self):
        p = {name: "x" for name in profile.REQUIRED}
        p["skills"] = []
        assert "skills" in profile.missing(p)


class TestRender:
    """The tool's reply is the interviewer's checklist."""

    def test_it_names_what_is_still_missing(self):
        out = profile.render({"full_name": "Sam"})
        assert "STILL MISSING" in out
        assert "work_setup" in out
        # And what is already known, so it does not ask twice.
        assert "Sam" in out

    def test_an_empty_profile_says_so_plainly(self):
        """Rather than an empty list, which reads as a broken tool."""
        assert "nothing recorded yet" in profile.render({})

    def test_completion_is_announced_as_an_instruction(self):
        """The model has to KNOW the interview is over, and what to do then.

        Reporting "missing: none" leaves it to infer that it should stop, and a
        model that is unsure keeps asking questions.
        """
        out = profile.render({name: "x" for name in profile.REQUIRED})
        assert "ALL REQUIRED FIELDS ARE NOW FILLED" in out
        assert "thank them" in out
        assert "STILL MISSING" not in out

    def test_lists_are_rendered_as_prose_not_json(self):
        """A model reads lines back as facts and JSON back as a structure to echo."""
        out = profile.render({"skills": ["Python", "Go"]})
        assert "Python, Go" in out
        assert "[" not in out


class TestDeclaration:
    """The tool spec is generated from FIELDS, so the two cannot drift."""

    def test_every_field_is_declared(self):
        properties = profile.declaration()["parameters"]["properties"]
        assert set(properties) == {f["name"] for f in profile.FIELDS}

    def test_nothing_is_mandatory_in_the_schema(self):
        """Even the required fields.

        The model records what it has so far and calls again as it learns more.
        A schema demanding all of them would force it to either invent the
        missing ones or never call the tool at all.
        """
        assert profile.declaration()["parameters"]["required"] == []

    def test_array_fields_declare_their_item_type(self):
        """Omitting `items` is not a tolerated default.

        The API rejects the whole setup message, so the session never opens --
        the entire interview mode is dead rather than one tool being broken.
        Found exactly that way:
          function_declarations[1].parameters.properties[interests].items:
          missing field
        """
        properties = profile.declaration()["parameters"]["properties"]
        for field in profile.FIELDS:
            if field.get("type") == "ARRAY":
                assert "items" in properties[field["name"]], field["name"]

    def test_the_description_tells_the_model_when_to_call(self):
        """Called as things are learned, not at the end.

        Calling it once at the end makes it a transcription step; calling it as
        it goes makes its result the checklist.
        """
        assert "as soon as you learn" in profile.declaration()["description"]


@pytest.mark.parametrize("name", ["skills", "interests"])
def test_the_list_fields_are_lists(name):
    field = next(f for f in profile.FIELDS if f["name"] == name)
    assert field.get("type") == "ARRAY"
