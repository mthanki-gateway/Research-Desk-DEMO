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


class TestNotes:
    """The difference between a profile and a spreadsheet.

    "work_setup: hybrid" is true and nearly useless. "hybrid -- firm about it,
    mentioned a long commute, sounded like he had negotiated it before" is the
    same answer with the part that matters still attached. All of that is
    present when the model hears it and gone by the time anyone reads a table.

    WHY THE PARSING IS TOLERANT. The tool declares notes as {field, note}
    objects, and the API does not enforce an array's item schema -- so whatever
    the model produces is what arrives. Observed in one afternoon: the declared
    shape, `observation` instead of `note`, and bare strings with no object at
    all. Reading only the declared key cost every note in a full interview:
    they were recorded, the tool reported success, and the stored profile came
    back empty.
    """

    def test_the_declared_shape(self):
        p = profile.merge({}, {"notes": [{"field": "skills", "note": "lit up about Terraform"}]})
        assert p["notes"] == {"skills": ["lit up about Terraform"]}

    def test_observation_instead_of_note(self):
        """Seen from the model despite the schema saying `note`."""
        p = profile.merge(
            {}, {"notes": [{"field": "work_setup", "observation": "firm; long commute"}]}
        )
        assert p["notes"] == {"work_setup": ["firm; long commute"]}

    def test_a_bare_string_is_kept_as_a_general_note(self):
        """Losing the field it belonged to is a small harm. Losing the
        observation is the whole feature."""
        p = profile.merge({}, {"notes": ["hesitated over the number"]})
        assert p["notes"] == {"general": ["hesitated over the number"]}

    def test_a_field_keyed_object_is_read_as_field_and_note(self):
        p = profile.merge({}, {"notes": [{"years_experience": "seemed unsure"}]})
        assert p["notes"] == {"years_experience": ["seemed unsure"]}

    def test_notes_accumulate_across_calls(self):
        p = profile.merge({}, {"notes": [{"field": "skills", "note": "enjoys Terraform"}]})
        p = profile.merge(p, {"notes": [{"field": "skills", "note": "rusty on Kubernetes"}]})
        assert p["notes"]["skills"] == ["enjoys Terraform", "rusty on Kubernetes"]

    def test_a_repeated_note_is_not_stored_twice(self):
        """The model re-sends what it already recorded when it calls again."""
        note = [{"field": "skills", "note": "enjoys Terraform"}]
        p = profile.merge({}, {"notes": note})
        p = profile.merge(p, {"notes": note})
        assert p["notes"]["skills"] == ["enjoys Terraform"]

    def test_an_unknown_field_is_filed_not_dropped(self):
        """It goes to `other` -- see TestQuotes for why that is the right bucket."""
        p = profile.merge({}, {"notes": [{"field": "vibe", "note": "relaxed"}]})
        assert p["notes"] == {"other": ["relaxed"]}

    def test_an_empty_note_is_ignored(self):
        p = profile.merge({}, {"notes": [{"field": "skills", "note": "  "}, ""]})
        assert p.get("notes") in ({}, None)

    def test_notes_do_not_disturb_the_fields(self):
        p = profile.merge(
            {},
            {"skills": ["Python"], "notes": [{"field": "skills", "note": "confident"}]},
        )
        assert p["skills"] == ["Python"]
        assert p["notes"] == {"skills": ["confident"]}

    def test_notes_are_never_required(self):
        """An interview that will not finish until every answer has an
        observation attached is one that invents observations."""
        full = {name: "x" for name in profile.REQUIRED}
        assert profile.complete(full)

    def test_they_are_rendered_under_their_field(self):
        """So the model can see what it already observed and not repeat it."""
        p = profile.merge(
            {},
            {"work_setup": "hybrid", "notes": [{"field": "work_setup", "note": "long commute"}]},
        )
        out = profile.render(p)
        assert "work_setup: hybrid" in out
        assert "long commute" in out
        assert out.index("work_setup: hybrid") < out.index("long commute")


class TestQuotes:
    """Their own words, kept alongside our reading of them.

    A quote survives every summary anyone writes later, and a reader trusts it
    in a way they never quite trust a paraphrase. It also happens to be the
    most reliable record of what was said: Gemini's `input_transcription` is a
    separate, lossier pass than the model's own understanding -- observed
    rendering a whole answer as "When the maintenance yesterday", and omitting
    a name the model then used correctly in its reply.
    """

    def test_quotes_are_stored_separately_from_notes(self):
        p = profile.merge(
            {},
            {
                "notes": [{"field": "work_setup", "note": "firm about it"}],
                "quotes": [{"field": "work_setup", "quote": "Not going back to an office."}],
            },
        )
        assert p["notes"] == {"work_setup": ["firm about it"]}
        assert p["quotes"] == {"work_setup": ["Not going back to an office."]}

    def test_quotes_accept_the_loose_shapes_too(self):
        """Same reasoning as notes: the API does not enforce item schemas."""
        p = profile.merge({}, {"quotes": ["I left over it."]})
        assert p["quotes"] == {"general": ["I left over it."]}

    def test_quotes_accumulate_and_dedupe(self):
        q = [{"field": "skills", "quote": "Terraform is the fun part."}]
        p = profile.merge({}, {"quotes": q})
        p = profile.merge(p, {"quotes": q})
        assert p["quotes"]["skills"] == ["Terraform is the fun part."]

    def test_an_unattachable_note_goes_to_other_not_general(self):
        """`other` is where the interesting half of an interview ends up.

        The fields were chosen in advance and the person was not, so a note
        that fits no field is not a stray -- it is usually the point.
        """
        p = profile.merge({}, {"notes": [{"field": "hobbies", "note": "restores motorbikes"}]})
        assert p["notes"] == {"other": ["restores motorbikes"]}


class TestEnding:
    """The model decides when the conversation is over.

    SEPARATE FROM COMPLETENESS, deliberately. "Every required field is filled"
    and "this interview is finished" are different facts: the interviewer fills
    the last field, then asks whether there is anything to add -- and that
    answer is frequently the most useful thing in the profile, because it is
    the only part the participant chose. Ending on completeness would cut the
    conversation off exactly there.
    """

    def test_the_end_tool_is_declared(self):
        assert profile.end_declaration()["name"] == profile.END_TOOL

    def test_it_asks_for_a_summary_but_does_not_demand_one(self):
        """A model with nothing to say should still be able to end."""
        params = profile.end_declaration()["parameters"]
        assert "summary" in params["properties"]
        assert params["required"] == []

    def test_it_says_when_not_to_call(self):
        """Called early, it closes the microphone mid-question."""
        text = profile.end_declaration()["description"]
        assert "do not call it while you are still" in text


class TestBucketValidation:
    """An invented bucket must not become a section of the card.

    The fallback that recovers a note from {field_name: text} can REASSIGN the
    field, so validating before it runs lets a made-up name through. Observed:
    a quote sent as {"note": "..."} was filed under a bucket literally called
    "note" instead of landing in `other`.
    """

    def test_a_made_up_bucket_is_corrected(self):
        p = profile.merge({}, {"quotes": [{"note": "two days is about right"}]})
        assert p["quotes"] == {"other": ["two days is about right"]}

    def test_a_real_field_survives_the_fallback(self):
        p = profile.merge({}, {"quotes": [{"field": "skills", "quote": "Terraform"}]})
        assert p["quotes"] == {"skills": ["Terraform"]}

    @pytest.mark.parametrize("bucket", ["general", "other"])
    def test_the_extra_buckets_are_allowed(self, bucket):
        p = profile.merge({}, {"notes": [{"field": bucket, "note": "x"}]})
        assert list(p["notes"]) == [bucket]


class TestDensity:
    """A thin profile is the failure mode, not a missing field.

    The required fields fill up early and then stop moving, so nothing about
    them tells the model that it is recording answers and discarding the
    person. The tool result counts what has been gathered and says so.
    """

    def test_the_counts_are_reported(self):
        p = profile.merge(
            {},
            {
                "notes": [{"field": "skills", "note": "a"}],
                "quotes": [{"field": "skills", "quote": "b"}],
            },
        )
        assert "1 note(s), 1 quote(s)" in profile.render(p)

    def test_a_thin_profile_is_called_thin(self):
        assert "THIN" in profile.render({"full_name": "Sam"})

    def test_a_rich_profile_is_not_nagged(self):
        notes = [{"field": "general", "note": f"observation {i}"} for i in range(8)]
        p = profile.merge({}, {"notes": notes})
        assert "THIN" not in profile.render(p)


class TestInventedFields:
    """The model can pass any key it likes. None of them become a column.

    `salary_expectation` and `favourite_language` used to be stored verbatim as
    profile keys -- and the card renders only the DECLARED fields, so they were
    written to the database and then invisible to everyone. The worst of both:
    the information was captured and could not be read.
    """

    def test_an_unknown_key_becomes_an_other_note(self):
        p = profile.merge({}, {"salary_expectation": "120k"})
        assert "salary_expectation" not in p
        assert p["notes"]["other"] == ["salary expectation: 120k"]

    def test_an_unknown_list_is_flattened_into_the_note(self):
        p = profile.merge({}, {"favourite_languages": ["Rust", "Go"]})
        assert p["notes"]["other"] == ["favourite languages: Rust, Go"]

    def test_declared_fields_are_untouched(self):
        p = profile.merge({}, {"full_name": "John", "location": "Ahmedabad"})
        assert p["full_name"] == "John"
        assert p["location"] == "Ahmedabad"
        assert "notes" not in p

    def test_a_real_field_and_an_invented_one_in_one_call(self):
        p = profile.merge({}, {"full_name": "John", "salary_expectation": "120k"})
        assert p["full_name"] == "John"
        assert p["notes"]["other"] == ["salary expectation: 120k"]


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
        every = {f["name"]: "x" for f in profile.FIELDS}
        out = profile.render(every)
        assert "ALL FIELDS ARE NOW FILLED" in out
        assert "thank them" in out
        assert "STILL MISSING" not in out

    def test_empty_optionals_are_named_once_the_required_ones_are_done(self):
        """Otherwise they are never asked for at all.

        They existed only in one line of the prompt, and nothing in the tool
        result mentioned them -- so `location` went unfilled in every
        interview. Naming them here puts them in front of the model at the
        moment it is choosing what to ask next.
        """
        out = profile.render({name: "x" for name in profile.REQUIRED})
        assert "still empty" in out
        assert "location" in out

    def test_a_declined_optional_is_not_a_reason_to_continue(self):
        """The nudge must not become a loop.

        Somebody who will not say where they live is not going to say it the
        fourth time either, and an interview that cannot end is worse than one
        missing a field.
        """
        out = profile.render({name: "x" for name in profile.REQUIRED})
        assert "leave it and finish" in out

    def test_optionals_are_also_named_while_required_ones_remain(self):
        """As a secondary list, so they can be picked up in passing."""
        out = profile.render({"full_name": "Sam"})
        assert "STILL MISSING" in out
        assert "worth asking if it fits" in out

    def test_lists_are_rendered_as_prose_not_json(self):
        """A model reads lines back as facts and JSON back as a structure to echo."""
        out = profile.render({"skills": ["Python", "Go"]})
        assert "Python, Go" in out
        assert "[" not in out


class TestDeclaration:
    """The tool spec is generated from FIELDS, so the two cannot drift."""

    def test_every_field_is_declared(self):
        """Plus `notes`, which is a channel alongside the fields, not one of them."""
        properties = profile.declaration()["parameters"]["properties"]
        assert set(properties) == {f["name"] for f in profile.FIELDS} | {
            profile.NOTES_KEY,
            profile.QUOTES_KEY,
        }

    def test_notes_are_declared_as_an_array_of_objects(self):
        """The shape the model is ASKED for, even though it sends others.

        Declaring it loosely would invite the loose shapes; declaring it
        precisely and parsing tolerantly gets the structured form most of the
        time and loses nothing the rest of the time.
        """
        notes = profile.declaration()["parameters"]["properties"][profile.NOTES_KEY]
        assert notes["type"] == "ARRAY"
        assert set(notes["items"]["properties"]) == {"field", "note"}

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


class TestPersonName:
    """Finding the interviewee's name in a schema nobody wrote in advance.

    Interview has a fixed `full_name`. Howler's fields come from a brief, so
    the name arrives as `candidate_name`, `respondent_name`, or not at all --
    which is why a finished Howler conversation stayed titled "Interview" and
    its link stayed "Unnamed" next to a profile that plainly knew who it was.
    """

    def test_it_finds_the_fixed_field(self):
        assert profile.person_name({"full_name": "John Doe"}) == "John Doe"

    def test_it_finds_a_generated_one(self):
        assert profile.person_name({"candidate_name": "Priya Raman"}) == "Priya Raman"

    def test_a_list_is_flattened(self):
        """ARRAY fields are legal, and the model sometimes picks one."""
        assert profile.person_name({"respondent_name": ["Sam Okoro"]}) == "Sam Okoro"

    def test_a_company_is_not_a_person(self):
        """The reason the lookup is an allowlist rather than a `*_name` match.

        Naming somebody's interview after their employer is worse than leaving
        it unnamed, because it looks deliberate.
        """
        assert profile.person_name({"company_name": "Acme Logistics"}) == ""
        assert profile.person_name({"product_name": "Acme Cloud"}) == ""

    def test_a_missing_name_is_not_the_string_None(self):
        """`_flat` is a formatter, not a validator.

        Handed a missing field it returned the literal "None", which passed
        every check after it and retitled the conversation "None".
        """
        assert profile.person_name({}) == ""
        assert profile.person_name({"full_name": None}) == ""
        assert profile.person_name({"current_role": "platform engineer"}) == ""

    def test_whitespace_is_not_a_name(self):
        assert profile.person_name({"full_name": "   "}) == ""

    def test_an_introduction_is_not_a_name(self):
        """A whole sentence filed under a name field is a summary, not a title."""
        long = "John Doe, a senior engineer looking for remote work in Berlin"
        assert profile.person_name({"full_name": long}) == ""
