import argparse
import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
import json
from pathlib import Path
from agents.interpretation_agent import InterpretationResult
from agents.structure_agent import (
    StructureAgent,
    StructureError,
    StructureResult,
)
from configs.settings import TRANSLATION_MODEL
from models.model_factory import ModelFactory


class FakeModel:
    """Return a canned response without calling the real model."""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None
        self.last_kwargs = None

    async def generate(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


def make_interpretation(**overrides) -> InterpretationResult:
    """Build a valid interpretation with sensible defaults."""
    values = {
        "utterance_type": "request",
        "actor": "the caregiver",
        "action": "give",
        "object": "grandmother's medicine",
        "timing": "after dinner",
        "negated": False,
        "urgency": "normal",
    }
    values.update(overrides)
    return InterpretationResult(**values)


class StructureAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_response_is_parsed_correctly(self):
        fake = FakeModel(
            '{"draft_translation": '
            '"Give grandmother her medicine after dinner."}'
        )
        agent = StructureAgent(model=fake)

        result = await agent.run(make_interpretation())

        self.assertEqual(
            result,
            StructureResult(
                draft_translation=(
                    "Give grandmother her medicine after dinner."
                )
            ),
        )

    async def test_response_wrapped_in_markdown_fences_still_parses(self):
        fake = FakeModel(
            '```json\n'
            '{"draft_translation": "Do not give her the blue pill."}'
            '\n```'
        )
        agent = StructureAgent(model=fake)

        result = await agent.run(
            make_interpretation(
                object="the blue pill",
                negated=True,
                negation_cue="do not",
                negation_scope="give the blue pill",
            )
        )

        self.assertEqual(
            result.draft_translation,
            "Do not give her the blue pill.",
        )

    async def test_dictionary_model_response_is_supported(self):
        fake = FakeModel({
            "text": (
                '{"draft_translation": '
                '"Please call the nurse immediately."}'
            )
        })
        agent = StructureAgent(model=fake)

        result = await agent.run(
            make_interpretation(
                action="call",
                object="the nurse",
                timing="immediately",
                urgency="high",
            )
        )

        self.assertEqual(
            result.draft_translation,
            "Please call the nurse immediately.",
        )

    async def test_unparseable_response_raises_structure_error(self):
        fake = FakeModel("this is not JSON")
        agent = StructureAgent(model=fake)

        with self.assertRaisesRegex(
            StructureError,
            "Invalid structure-agent output",
        ):
            await agent.run(make_interpretation())

    async def test_missing_draft_translation_raises_structure_error(self):
        fake = FakeModel('{"translation": "Wrong field name"}')
        agent = StructureAgent(model=fake)

        with self.assertRaises(StructureError):
            await agent.run(make_interpretation())

    async def test_draft_translation_must_be_a_string(self):
        fake = FakeModel('{"draft_translation": ["not", "a", "string"]}')
        agent = StructureAgent(model=fake)

        with self.assertRaises(StructureError):
            await agent.run(make_interpretation())

    async def test_interpretation_and_recent_context_are_added_to_prompt(self):
        fake = FakeModel(
            '{"draft_translation": '
            '"Give grandmother her medicine after dinner."}'
        )
        agent = StructureAgent(model=fake)
        interpretation = make_interpretation()
        recent_context = [
            {
                "speaker": "caregiver",
                "text": "Her medicine is on the table.",
            }
        ]

        await agent.run(
            interpretation=interpretation,
            recent_context=recent_context,
        )

        self.assertIn('"action": "give"', fake.last_prompt)
        self.assertIn('"timing": "after dinner"', fake.last_prompt)
        self.assertIn(
            "Her medicine is on the table.",
            fake.last_prompt,
        )

    async def test_empty_recent_context_is_serialized_as_empty_list(self):
        fake = FakeModel('{"draft_translation": "She needs to rest."}')
        agent = StructureAgent(model=fake)

        await agent.run(
            make_interpretation(
                utterance_type="statement",
                actor="grandmother",
                action="rest",
                object=None,
                timing=None,
            )
        )

        self.assertIn("Recent context:\n[]", fake.last_prompt)

    async def test_verification_feedback_is_added_to_retry_prompt(self):
        fake = FakeModel('{"draft_translation": "Give it after dinner."}')
        agent = StructureAgent(model=fake)

        await agent.run(
            make_interpretation(),
            verification_issues=["The draft omitted the timing."],
        )

        self.assertIn(
            '["The draft omitted the timing."]',
            fake.last_prompt,
        )

    async def test_model_is_called_with_deterministic_settings(self):
        fake = FakeModel(
            '{"draft_translation": '
            '"Give grandmother her medicine after dinner."}'
        )
        agent = StructureAgent(model=fake)

        await agent.run(make_interpretation())

        self.assertEqual(fake.last_kwargs["temperature"], 0)
        self.assertEqual(fake.last_kwargs["max_tokens"], 300)

    


def check_case(case, result):
    text = result.draft_translation.casefold()
    rules = case.get("assert", {})
    failures = []

    for phrase in rules.get("required_phrases", []):
        if phrase.casefold() not in text:
            failures.append(f"missing required phrase: {phrase!r}")

    for key in ("required_any_phrases", "required_any_urgency_phrases"):
        choices = rules.get(key, [])
        if choices and not any(choice.casefold() in text for choice in choices):
            failures.append(f"none of {key} found: {choices!r}")

    for phrase in rules.get("forbidden_phrases", []):
        if phrase.casefold() in text:
            failures.append(f"found forbidden phrase: {phrase!r}")

    return failures


async def evaluate(args):
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    agent = StructureAgent(model=ModelFactory.create(args.model))
    records = []

    for case in cases:
        started = time.perf_counter()
        try:
            result = await agent.run(
                InterpretationResult(**case["interpretation"]),
                None if args.no_context else case.get("recent_context"),
            )
            failures = check_case(case, result)
            status = "PASS" if not failures else "FAIL"
            output = result.model_dump()
        except StructureError as error:
            status = "ERROR"
            failures = [str(error)]
            output = None

        record = {
            "id": case["id"],
            "probe": case.get("probe", ""),
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "failures": failures,
            "result": output,
        }
        records.append(record)
        print(f"{status:<5} {case['id']}  {record['latency_ms']:>5} ms  {record['probe']}")
        for failure in failures:
            print(f"      - {failure}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    passed = sum(record["status"] == "PASS" for record in records)
    print(f"\n{passed}/{len(records)} passed  results -> {args.results}")
    return 0 if passed == len(records) else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Structure Agent")
    parser.add_argument(
        "--cases", type=Path,
        default=Path(__file__).parent / "structure_test_cases.json",
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--model", default=TRANSLATION_MODEL)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    if args.results is None:
        filename = (
            "results_baseline_structure.json"
            if args.no_context
            else "results_full_structure.json"
        )
        args.results = Path("tests") / filename
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    sys.exit(main())
