"""
agents/verify_agent.py

Follows the same pattern as agents/base_agent_example.py:
- subclasses BaseAgent
- gets its model via ModelFactory (not boto3 directly)
- implements run()

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

# schema
class VerificationResult(PydanticModel):
    verdict: Literal["PASS", "RETRY", "CLARIFY"]
    issues: list[str] = Field(default_factory=list)
    fault_source: Literal["interpretation", "structure", "source_ambiguity"] | None = None
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
 
You are given the original transcript, the extracted interpretation (a
structured object with fields including actor, action, object, timing,
negated, negation_cue, negation_scope, urgency, and ambiguity), and the
draft translation. Judge whether critical meaning survived, and if not,
diagnose which stage is responsible.

Trust the interpretation's resolved fields as given. The Interpretation
Agent may have access to conversation history or person info that you do
not see here - do not second-guess a resolved actor, object, or timing just
because it is not self-evident from the transcript alone. Your job is
comparing the draft against the interpretation, and the interpretation
against anything the transcript states unambiguously - not re-deriving
whether the interpretation's resolutions are independently justified.
 
Follow this order:
 
1. Does the draft accurately reflect the interpretation? Compare against
   negation_scope specifically: if negated is true, only the item named in
   negation_scope is negated - NOT the entire action. A draft that gives an
   affirmative instruction for the non-negated part while correctly negating
   only what negation_scope names is CORRECT, not a fault. A different
   sentence structure (e.g. passive voice) is not itself a fault - only flag
   a difference that changes WHAT was communicated (actor, action, object,
   timing, negation_scope, urgency), never HOW it was phrased.
   If the draft misrepresents any of these relative to the interpretation:
   verdict = "RETRY", fault_source = "structure".
 
2. If the draft matches the interpretation, does the interpretation itself
   contradict or misread something the TRANSCRIPT states clearly and
   unambiguously (not vague, actually stated)? If so, this is an
   INTERPRETATION fault: verdict = "RETRY", fault_source = "interpretation".
 
3. The interpretation's ambiguity field is a list of specific unresolved
   issues, populated directly by the Interpretation Agent - it is not
   something you need to infer from wording elsewhere. If ambiguity is
   non-empty, that signals genuine SOURCE AMBIGUITY: verdict = "CLARIFY",
   fault_source = "source_ambiguity", regardless of how confidently other
   fields are filled in. Do not return RETRY here - retrying cannot resolve
   an ambiguity that exists in the source.
 
4. Otherwise: verdict = "PASS".
 
Respond with ONLY a JSON object, no other text, no markdown fences, matching
exactly this shape:
{"verdict": "PASS" | "RETRY" | "CLARIFY", "issues": [string, ...], "fault_source": "interpretation" | "structure" | "source_ambiguity" | null, "clarification_question": string | null}"""
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

        raw_text = await self.ask_model(prompt, temperature=0)
        return self._parse_result(raw_text)

    # safety net: if the model returns unparseable text, return a CLARIFY with an issue and a generic clarification question
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
                "utterance_type": "request",
                "actor": "FDW",
                "action": "give medicine",
                "object": "ah ma",
                "timing": "after dinner",
                "negated": True,
                "negation_cue": "not",
                "negation_scope": "before dinner",
                "urgency": "normal",
                "ambiguity": [],
                "clarification_question": None,
            },
            "draft_translation": "Please give grandmother her medicine before dinner.",
        })
        print(result.model_dump_json(indent=2))
 
    asyncio.run(main())