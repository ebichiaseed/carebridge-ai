import asyncio
import unittest

from agents.verify_agent import VerifyAgent, VerificationResult


class FakeModel:
    """Same style as FakeSpeechModel in test_transcription_service.py -
    returns a canned response instead of calling real Bedrock."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0)
        return self.response_text


class VerifyAgentTests(unittest.TestCase):
    def test_pass_verdict_is_parsed_correctly(self): # test 1: PASS case
        fake = FakeModel(
            '{"verdict": "PASS", "issues": [], "fault_source": null, '
            '"clarification_question": null}'
        )
        agent = VerifyAgent(model=fake)

        result = asyncio.run(agent.run({
            "transcript": "Give ah ma her medicine after dinner.",
            "interpretation": {"timing": "after dinner"},
            "draft_translation": "Give grandmother her medicine after dinner.",
        }))

        self.assertEqual(result, VerificationResult(verdict="PASS"))

    def test_retry_verdict_flags_translation_fault(self): # test 2: RETRY case
        fake = FakeModel(
            '{"verdict": "RETRY", "issues": ["timing was flipped"], '
            '"fault_source": "translation", "clarification_question": null}'
        )
        agent = VerifyAgent(model=fake)

        result = asyncio.run(agent.run({
            "transcript": "Give ah ma her medicine after dinner, not before.",
            "interpretation": {"timing": "after dinner", "negated": True},
            "draft_translation": "Give grandmother her medicine before dinner.",
        }))

        self.assertEqual(result.verdict, "RETRY")
        self.assertEqual(result.fault_source, "translation")

    def test_unparseable_response_fails_safe_to_clarify(self): # test 3: UNPARSEABLE case
        fake = FakeModel("this is not json at all")
        agent = VerifyAgent(model=fake)

        result = asyncio.run(agent.run({
            "transcript": "...",
            "interpretation": {},
            "draft_translation": "...",
        }))

        # Never crash, never silently PASS on a broken response.
        self.assertEqual(result.verdict, "CLARIFY")
        self.assertTrue(len(result.issues) > 0)

    def test_response_wrapped_in_markdown_fences_still_parses(self): # test 4: MARKDOWN FENCES case
        fake = FakeModel(
            '```json\n{"verdict": "PASS", "issues": [], '
            '"fault_source": null, "clarification_question": null}\n```'
        )
        agent = VerifyAgent(model=fake)

        result = asyncio.run(agent.run({
            "transcript": "...",
            "interpretation": {},
            "draft_translation": "...",
        }))

        self.assertEqual(result.verdict, "PASS")


if __name__ == "__main__":
    unittest.main()
