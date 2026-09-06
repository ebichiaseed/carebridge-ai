# agents/display_agent.py
"""
Display translation: renders finished output in the language the reader needs.

This runs AFTER the graph, not inside it. The interpretation, structure and
verify agents all reason in English, and verification compares an English draft
against the transcript -- that is what makes Critical Meaning Preservation and
Verifier Catch Rate measurable. Translating inside the graph would mean the
verifier had to check meaning across a language boundary as well as across a
rewrite, which is a harder job and a weaker signal.

So English stays canonical. This agent is a presentation layer:

    graph output (English)  ->  DisplayAgent  ->  what the screen shows

Two fields, two different target languages:

- translation goes to the LISTENER, so it is rendered in the requested target
  language.
- clarification_question goes back to the SPEAKER, so it is rendered in the
  language they spoke.

Nothing here feeds back into the pipeline, and the English text is kept
alongside so run logs and evaluation stay in one language.

Instantiate at app startup:
    agent = DisplayAgent(model=ModelFactory.create("haiku"))
"""

from pydantic import BaseModel

from agents.base_agent import BaseAgent


# Codes this agent renders into. "singlish" and "mix" are input-only labels and
# are never targets: both render as English.
LANGUAGE_NAMES = {
    "english": "English",
    "chinese": "Simplified Chinese",
}

DEFAULT_LANGUAGE = "english"


class DisplayResult(BaseModel):
    translation: str | None = None
    clarification_question: str | None = None


SYSTEM_PROMPT = """\
You are the display stage of CareBridge, a caregiving translation tool used in \
Singaporean households.

You are given finished English text that has already been checked for accuracy.
Your only job is to render it in the requested language. You are not
interpreting, not summarising and not improving it.

Rules:
- Preserve meaning exactly. Actor, action, object, timing, negation and urgency
  must all survive. A dropped "not" is the most harmful error you can make.
- Keep it plain and spoken, the way one household member speaks to another.
  Short sentences. No formal or medical register unless the English used one.
- Keep proper names, brand names and dosages unchanged.
- If a field's target language is English, return that field exactly as given,
  character for character.
- If a field is null, return null for it.

Return ONLY a JSON object. No preamble, no markdown fences, no commentary.

{
  "translation": string or null,
  "clarification_question": string or null
}
"""


class DisplayAgent(BaseAgent):

    def __init__(self, model, tools=None):
        super().__init__(
            name="DISPLAY_AGENT",
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    async def run(
        self,
        translation: str | None,
        clarification_question: str | None = None,
        target_language: str = DEFAULT_LANGUAGE,
        question_language: str | None = None,
    ) -> DisplayResult:
        """Render finished English text for display.

        target_language:   language for `translation` (what the listener reads).
        question_language: language for `clarification_question` (what the
                           speaker reads). Defaults to target_language.
        """

        target = LANGUAGE_NAMES.get(target_language, LANGUAGE_NAMES[DEFAULT_LANGUAGE])
        question_target = LANGUAGE_NAMES.get(
            question_language or target_language, target
        )

        original = DisplayResult(
            translation=translation,
            clarification_question=clarification_question,
        )

        # Nothing to render, or nothing to change. Skip the model call entirely:
        # this is the common case and it should cost nothing.
        if translation is None and clarification_question is None:
            return original
        if target == "English" and question_target == "English":
            return original

        prompt = f"""\
TRANSLATION (render in {target}):
{translation if translation is not None else "null"}

CLARIFICATION QUESTION (render in {question_target}):
{clarification_question if clarification_question is not None else "null"}
"""

        try:
            raw = await self.ask_model(prompt, temperature=0, max_tokens=600)
            text = raw["text"] if isinstance(raw, dict) else raw
            result = DisplayResult.model_validate_json(_extract_json(text))
        except Exception as error:
            # Display is cosmetic. A failure here must never lose the verified
            # English text -- fall back to it and let the caller log the error.
            raise DisplayError(str(error)) from error

        # A null coming back for text that was not null means the model dropped
        # a field. Keep the English rather than showing an empty panel.
        return DisplayResult(
            translation=(
                result.translation if result.translation is not None else translation
            ),
            clarification_question=(
                result.clarification_question
                if result.clarification_question is not None
                else clarification_question
            ),
        )


class DisplayError(Exception):
    """Rendering failed. The caller should fall back to the English text."""


def _extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return text[start:end + 1]


def to_trace(
    target_language: str,
    question_language: str | None,
    rendered: bool,
    schema_valid: bool = True,
) -> dict:
    """Trace fields for the run log. No text, only what was asked for."""
    return {
        "node": "display",
        "schema_valid": schema_valid,
        "target_language": target_language,
        "question_language": question_language or target_language,
        # False means the model call was skipped because English was requested.
        "model_called": rendered,
    }