# agents/interpretation_agent.py
"""
Interpretation node: turns a raw transcript into a structured InterpretationResult.

Does NOT call the glossary itself -- glossary retrieval is a separate upstream node
(see graph flow step 4). Hits are passed in via `glossary_hits`.

Instantiate at app startup, not on import:
    agent = InterpretationAgent(model=ModelFactory.create("haiku"))
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent


class InterpretationResult(BaseModel):
    """Shared contract -- import this, do not redefine it per node."""
    actor: str | None = None
    action: str
    object: str | None = None
    timing: str | None = None
    negated: bool = False
    urgency: Literal["low", "normal", "high"] = "normal"
    ambiguity: list[str] = Field(default_factory=list)
    # CareBridge additions
    source_language: str | None = None
    cleaned_transcript: str | None = None


class InterpretationError(Exception):
    """Raised when the model output cannot be validated. Node should route to CLARIFY."""


SYSTEM_PROMPT = """\
You are the interpretation stage of a voice assistant for Singaporean caregivers and \
elderly users who speak Singlish (English mixed with Hokkien, Malay and Mandarin particles).

Your job is to extract the STRUCTURE of what was said, not to translate it.

You are given four inputs and they resolve different things:
- GLOSSARY TERMS: what a dialect word means. Use these meanings; do not guess your own.
- PERSON INFO: who people and objects refer to. Use it to resolve "ah girl", "my pill", "auntie".
- RECENT CONTEXT: earlier turns. Use it to resolve pronouns, "that one", "same as just now".
- TRANSCRIPT: the raw speech-to-text output, which may contain recognition errors.

Rules:
- negated MUST be true if the utterance contains any negation (don't, cannot, no need,
  bo, mai, buay). Getting this wrong is the most harmful error you can make.
- urgency is "high" only for pain, falls, breathing difficulty, or explicit distress.
- If a referent cannot be resolved from person info or context, leave that field null and
  add a short note to ambiguity. Do NOT invent a plausible referent.
- cleaned_transcript: the transcript with obvious recognition errors corrected. If you were
  given multiple candidate transcripts, reconcile them into one best reading, using the
  glossary as evidence for which candidate heard a dialect term correctly.

Return ONLY a JSON object. No preamble, no markdown fences, no trailing commentary.

{
  "actor": string or null,
  "action": string,
  "object": string or null,
  "timing": string or null,
  "negated": boolean,
  "urgency": "low" | "normal" | "high",
  "ambiguity": [string],
  "source_language": string or null,
  "cleaned_transcript": string
}
"""


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "(none matched)"
    return "\n".join(
        f"- {h['term']} ({h['language']}): {h['meaning']}"
        + (f' [e.g. "{h["example"]}"]' if h.get("example") else "")
        + (f" [ambiguity: {h['ambiguity']}]" if h.get("ambiguity") else "")
        for h in hits
    )


class InterpretationAgent(BaseAgent):

    def __init__(self, model, tools=None):
        super().__init__(
            name="INTERPRETATION_AGENT",
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    async def run(
        self,
        transcript: str | list[str],
        glossary_hits: list[dict] | None = None,
        recent_context: list[dict] | None = None,
        person_info: dict | None = None,
    ) -> InterpretationResult:

        if isinstance(transcript, list):
            transcript_block = "\n".join(
                f"candidate {i + 1}: {t}" for i, t in enumerate(transcript)
            )
        else:
            transcript_block = transcript

        prompt = f"""\
TRANSCRIPT:
{transcript_block}

GLOSSARY TERMS:
{_format_hits(glossary_hits or [])}

PERSON INFO:
{person_info or "(none)"}

RECENT CONTEXT:
{recent_context or "(none)"}
"""

        raw = await self.ask_model(prompt)
        text = raw["text"] if isinstance(raw, dict) else raw

        try:
            return InterpretationResult.model_validate_json(_strip_fences(text))
        except (ValidationError, ValueError) as error:
            # Do NOT fabricate a result. Let the graph route this to CLARIFY.
            raise InterpretationError(f"invalid interpretation output: {error}") from error


def _strip_fences(text: str) -> str:
    """Tolerate a fenced block if the model adds one anyway."""
    text = text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]
    return text