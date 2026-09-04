import json

from pydantic import BaseModel, ValidationError

from agents.base_agent import BaseAgent
from agents.interpretation_agent import InterpretationResult


class StructureResult(BaseModel):
    '''
    The output result for the StructureAgent to be ingested by the verification agent.
    '''
    draft_translation: str

class StructureError(Exception):
    pass

SYSTEM_PROMPT = """

You are the Structure Agent for CareBridge.

Convert the supplied interpretation into one or two clear, natural sentences.
Preserve the actor, action, object, timing, negation, and urgency exactly.
Do not invent information that is absent from the interpretation.

Return only JSON in this shape:
{"draft_translation": "string"}
"""
class StructureAgent(BaseAgent):

    def __init__(self, model, tools=None):
        super().__init__(
            name="STRUCTURE_AGENT",
            model=model,
            tools=tools,
            system_prompt= SYSTEM_PROMPT
        )


    async def run(self, 
                  interpretation: InterpretationResult,
                  recent_context: list[dict] | None = None) -> StructureResult:  

        prompt = (
            f"Interpretation:\n"
            f"{interpretation.model_dump_json(indent=2)}\n\n"
            f"Recent context:\n"
            f"{json.dumps(recent_context or [], ensure_ascii=False)}"
        )

        raw = await self.ask_model(prompt, temperature=0, max_tokens=300)
        text = raw["text"] if isinstance(raw, dict) else raw

        try:
            start = text.index("{")
            end = text.rindex("}") + 1

            return StructureResult.model_validate_json(
                text[start:end]
            )
        except (ValueError, ValidationError) as error:
            raise StructureError(
                f"Invalid structure-agent output: {error}"
            ) from error