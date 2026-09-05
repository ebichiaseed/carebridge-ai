"""
agents/verify_agent.py

Follows the same pattern as agents/base_agent_example.py:
- subclasses BaseAgent
- gets its model via ModelFactory (not boto3 directly)
- implements run()

Note: BedrockModel.generate() returns plain text, not a forced tool-call
JSON shape. So we ask Claude to respond in JSON only, then parse + validate
it ourselves with Pydantic. If parsing fails, we fail safe to CLARIFY
rather than crashing the graph - this matches the team's "no infinite
loop, no silent crash" invariant.
"""

import json
from typing import Literal

from pydantic import BaseModel as PydanticModel, Field, ValidationError

from agents.base_agent import BaseAgent
from models.model_factory import ModelFactory
from configs.settings import VERIFICATION_MODEL


# ---------------------------------------------------------------------------
# Shared contract - matches what A's routing code expects.
# If a schemas/ folder gets added later, this should move there so every
# agent can import the same definition instead of redefining it.
# ---------------------------------------------------------------------------

class VerificationResult(PydanticModel):
    verdict: Literal["PASS", "RETRY", "CLARIFY"]
    issues: list[str] = Field(default_factory=list)
    fault_source: Literal["translation", "source_ambiguity"] | None = None
    clarification_question: str | None = None


class VerifyAgent(BaseAgent):
    def __init__(self, model, tools=None):
        super().__init__(
            name="VerifyAgent",
            model=model,
            tools=tools,
            system_prompt="""You are the Verify Agent for CareBridge, a caregiving
translation tool between a Foreign Domestic Worker and an elderly Hokkien/
Cantonese-speaking employer.

Given the original transcript, the extracted interpretation, and the draft
translation, judge whether critical meaning survived translation. Check
specifically: actor, action, object, timing, negation, urgency.

- If meaning is preserved: verdict = "PASS".
- If the translation's wording/phrasing dropped or distorted meaning, but a
  re-draft could plausibly fix it: verdict = "RETRY", fault_source = "translation".
- If the transcript or interpretation itself is ambiguous (e.g. an unresolved
  "that one", unclear timing) such that no re-translation could fix it:
  verdict = "CLARIFY", fault_source = "source_ambiguity".

Respond with ONLY a JSON object, no other text, no markdown fences, matching
exactly this shape:
{"verdict": "PASS" | "RETRY" | "CLARIFY", "issues": [string, ...], "fault_source": "translation" | "source_ambiguity" | null, "clarification_question": string | null}"""
        )

    async def run(self, input_data: dict) -> VerificationResult:
        """
        input_data expects:
          {
            "transcript": str,
            "interpretation": dict,
            "draft_translation": str,
          }
        """
        prompt = (
            f"Original transcript: {input_data['transcript']}\n"
            f"Extracted interpretation: {json.dumps(input_data['interpretation'])}\n"
            f"Draft translation: {input_data['draft_translation']}"
        )

        raw_text = await self.ask_model(prompt)
        return self._parse_result(raw_text)

    def _parse_result(self, raw_text: str) -> VerificationResult:
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
            return VerificationResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Fail safe, never crash the graph and never invent a PASS.
            return VerificationResult(
                verdict="CLARIFY",
                issues=[f"Verifier returned an unparseable response: {e}"],
                clarification_question="Could you say that a different way?",
            )


# ---------------------------------------------------------------------------
# Quick manual check - not the real test, just a way to eyeball it works
# before wiring into the graph. Run: python -m agents.verify_agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        model = ModelFactory.create(VERIFICATION_MODEL)
        agent = VerifyAgent(model=model)
        result = await agent.run({
            "transcript": "Give ah ma her medicine after dinner, not before.",
            "interpretation": {
                "actor": "FDW",
                "action": "give medicine",
                "object": "ah ma (grandmother)",
                "timing": "after dinner",
                "negated": True,
                "urgency": "normal",
            },
            "draft_translation": "Please give grandmother her medicine before dinner.",
        })
        print(result.model_dump_json(indent=2))

    asyncio.run(main())
