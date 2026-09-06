import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent


class InterpretationResult(BaseModel):
    """Shared contract -- import this, do not redefine it per node."""

    # What was said
    utterance_type: Literal["request", "statement", "question", "distress"] = "statement"
    actor: str | None = None
    action: str | None = None         
    object: str | None = None
    timing: str | None = None
    negated: bool = False
    negation_cue: str | None = None   
    negation_scope: str | None = None  # what the negation applies to
    urgency: Literal["low", "normal", "high"] = "normal"
    ambiguity: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    source_language: str | None = None
    cleaned_transcript: str | None = None


class InterpretationError(Exception):
    """Raised when the model output cannot be validated. Node should route to CLARIFY."""
PROFILE_FIELDS = (
    "preferred_name",
    "languages",
    "household_terms",
    "relationships",
    "communication_preferences",
)


SYSTEM_PROMPT = """\
You are the interpretation stage of a voice assistant for Singaporean caregivers and \
elderly users who speak Singlish (English mixed with Hokkien, Malay and Mandarin particles).

Your job is to extract the STRUCTURE of what was said, not to translate it.

You are given five inputs and they resolve different things:
- GLOSSARY TERMS: what a dialect word means. Use these meanings; do not guess your own.
- PERSON INFO: who people and objects refer to. Use it to resolve "ah girl", "my pill", "auntie".
- RECENT CONTEXT: earlier turns. Use it to resolve pronouns, "that one", "same as just now".
- TRANSCRIPT: the raw speech-to-text output, which may contain recognition errors.
- VERIFICATION FEEDBACK: problems found in the previous draft. Reconsider the
  interpretation where those problems indicate that meaning was extracted incorrectly.
  Do not invent details solely to satisfy the feedback.

Rules on glossary terms:
- A term tagged (paraphrase match) was retrieved by similarity, not because the word
  appeared. Treat it as a candidate reading, not a confirmed meaning. If it does not
  fit the utterance, ignore it. A term you ignored must NOT produce an ambiguity note --
  discarding a bad candidate is not an unresolved reading.
- PERSON INFO resolves WHO someone is. It never resolves which SENSE of a term applies.
  Knowing that 阿嬤 is Mdm Tan does not settle whether 辛苦 means physically tired or
  unwell.
- If a term carries an [ambiguity: ...] note and neither RECENT CONTEXT nor the
  utterance itself settles which reading applies, add a short note to ambiguity saying
  which term is unresolved and what the competing readings are. Do this even when you
  have picked the more likely reading for the other fields.

Rules on meaning:
- negated MUST be true if the utterance contains any negation (don't, cannot, no need,
  bo, mai, buay). Getting this wrong is the most harmful error you can make.
  When negated is true, also set negation_cue to the exact trigger word and
  negation_scope to what is being negated.
- urgency is "normal" by default. Use "low" only when the speaker explicitly states
  something is not a concern. Never use "low" merely because nothing urgent was
  mentioned.
- urgency is "high" only for pain, falls, breathing difficulty, or explicit distress.
  Vague discomfort with no named symptom is NOT high -- flag it in ambiguity instead.
- action may be null when nothing is being asked for -- a plain remark or exclamation
  is utterance_type "statement" with a null action. Do not invent an action to fill it.
- If a referent cannot be resolved from person info or context, leave that field null
  and add a short note to ambiguity. Do NOT invent a plausible referent.
- Only flag a referent when knowing who or what it is changes what the listener must
  do. Do NOT flag:
  - a referent you resolved from RECENT CONTEXT or PERSON INFO. Once resolved,
    record it and move on. "Likely X but not explicitly confirmed" is a resolution,
    not an ambiguity -- do not add a note merely because the speaker did not say
    the full word;
  - the identity of a role referent (她, 阿嬤, ah ma) when the action, timing and
    urgency are clear without a name.
  DO flag when two or more distinct people or objects in context could be meant, or
  when acting on the wrong one would be harmful.
- clarification_question: if ambiguity is non-empty, write ONE short question, in the
  speaker's own register, that would resolve the most important unresolved item.
  Otherwise null.
- cleaned_transcript: the transcript with obvious recognition errors corrected. If you were
  given multiple candidate transcripts, reconcile them into one best reading, using the
  glossary as evidence for which candidate heard a dialect term correctly.

Return ONLY a JSON object. No preamble, no markdown fences, no trailing commentary.

{
  "utterance_type": "request" | "statement" | "question" | "distress",
  "actor": string or null,
  "action": string or null,
  "object": string or null,
  "timing": string or null,
  "negated": boolean,
  "negation_cue": string or null,
  "negation_scope": string or null,
  "urgency": "low" | "normal" | "high",
  "ambiguity": [string],
  "clarification_question": string or null,
  "source_language": string or null,
  "cleaned_transcript": string
}
"""


def _format_hits(hits: list[dict] | None) -> str:
    if not hits:
        return "(none matched)"

    lines = []
    for hit in hits:
        term = hit.get("term", "?")
        language = hit.get("language", "unknown")
        line = f"- {term} ({language}): {hit.get('meaning', '')}"
        if hit.get("match_type") == "vector":
            line += " (paraphrase match)"
        if hit.get("example"):
            line += f' [e.g. "{hit["example"]}"]'
        if hit.get("ambiguity"):
            line += f" [ambiguity: {hit['ambiguity']}]"
        lines.append(line)
    return "\n".join(lines)


def _format_person_info(person_info: dict | None) -> str:
    if not person_info:
        return "(none)"

    lines = [
        f"- {key}: {person_info[key]}"
        for key in PROFILE_FIELDS
        if person_info.get(key)
    ]
    return "\n".join(lines) if lines else "(none)"


def _format_context(recent_context: list[dict] | None) -> str:
    if not recent_context:
        return "(none)"
    return "\n".join(
        f"- {turn.get('speaker', 'unknown')}: {turn.get('text', '')}"
        for turn in recent_context
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
        verification_issues: list[str] | None = None,
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
{_format_hits(glossary_hits)}

PERSON INFO:
{_format_person_info(person_info)}

RECENT CONTEXT:
{_format_context(recent_context)}

VERIFICATION FEEDBACK:
{json.dumps(verification_issues or [], ensure_ascii=False)}
"""

        # temperature=0
        raw = await self.ask_model(prompt, temperature=0, max_tokens=800)
        text = raw["text"] if isinstance(raw, dict) else raw

        try:
            return InterpretationResult.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as error:
            raise InterpretationError(f"invalid interpretation output: {error}") from error


def needs_clarification(result: InterpretationResult) -> bool:
    if not result.ambiguity:
        return False
    if result.urgency == "high":
        return True
    if result.negated and result.negation_scope is None:
        # "don't give her the thing" with the thing unresolved -- the dangerous case
        return True
    if result.utterance_type in {"request", "distress"} and (
        result.action is None or result.object is None
    ):
        return True
    return False


def to_trace(result: InterpretationResult, schema_valid: bool = True) -> dict[str, Any]:
    return {
        "node": "interpret",
        "schema_valid": schema_valid,
        "utterance_type": result.utterance_type,
        "negated": result.negated,
        "urgency": result.urgency,
        "ambiguity_count": len(result.ambiguity),
        "needs_clarification": needs_clarification(result),
    }


def _extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return text[start:end + 1]
