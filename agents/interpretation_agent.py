# agents/interpretation_agent.py
"""
Interpretation node: turns a raw transcript into a structured InterpretationResult.

Does NOT call the glossary itself -- glossary retrieval is a separate upstream node
(see graph flow step 4). Hits are passed in via `glossary_hits`, in the shape
produced by glossary_lookup.GlossaryEntry.to_hit().

Instantiate at app startup, not on import:
    agent = InterpretationAgent(model=ModelFactory.create("haiku"))

SCHEMA NOTE FOR THE TEAM: InterpretationResult is a shared integration boundary.
Fields added since the first draft -- utterance_type, negation_cue,
negation_scope, clarification_question, and `action` becoming optional -- all have
defaults, so code that ignores them still works. Agree them before merging, and
move this model into schemas.py once that file exists.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent


class InterpretationResult(BaseModel):
    """Shared contract -- import this, do not redefine it per node."""

    # What was said
    utterance_type: Literal["request", "statement", "question", "distress"] = "statement"
    actor: str | None = None
    action: str | None = None          # optional: "aiyo, so hot today" has no action
    object: str | None = None
    timing: str | None = None

    # Negation, split out because a single bool cannot express
    # "don't give the BLUE pill, give the white one"
    negated: bool = False
    negation_cue: str | None = None    # the trigger word: "mai", "bo", "don't"
    negation_scope: str | None = None  # what the negation applies to

    urgency: Literal["low", "normal", "high"] = "normal"
    ambiguity: list[str] = Field(default_factory=list)
    clarification_question: str | None = None

    # CareBridge additions
    source_language: str | None = None
    cleaned_transcript: str | None = None


class InterpretationError(Exception):
    """Raised when the model output cannot be validated. Node should route to CLARIFY."""


# Only these keys are ever put in a prompt. The Definition of Done forbids
# sensitive profile data leaving the process, and a raw dict dump would include
# whatever anyone happened to store.
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

You are given four inputs and they resolve different things:
- GLOSSARY TERMS: what a dialect word means. Use these meanings; do not guess your own.
- PERSON INFO: who people and objects refer to. Use it to resolve "ah girl", "my pill", "auntie".
- RECENT CONTEXT: earlier turns. Use it to resolve pronouns, "that one", "same as just now".
- TRANSCRIPT: the raw speech-to-text output, which may contain recognition errors.

Rules on glossary terms:
- A term tagged (paraphrase match) was retrieved by similarity, not because the word
  appeared. Treat it as a candidate reading, not a confirmed meaning. If it does not
  fit the utterance, ignore it.
- If a term carries an [ambiguity: ...] note and neither RECENT CONTEXT nor PERSON INFO
  resolves which reading applies, add a short note to ambiguity saying which term is
  unresolved and what the competing readings are.

Rules on meaning:
- negated MUST be true if the utterance contains any negation (don't, cannot, no need,
  bo, mai, buay). Getting this wrong is the most harmful error you can make.
  When negated is true, also set negation_cue to the exact trigger word and
  negation_scope to what is being negated.
- urgency is "high" only for pain, falls, breathing difficulty, or explicit distress.
- action may be null when nothing is being asked for -- a plain remark or exclamation
  is utterance_type "statement" with a null action. Do not invent an action to fill it.
- If a referent cannot be resolved from person info or context, leave that field null and
  add a short note to ambiguity. Do NOT invent a plausible referent.
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
    """Render glossary_lookup hits for the prompt.

    Uses .get() throughout: as the glossary grows, one entry missing a field
    should degrade the line, not KeyError the whole node.
    """
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
    """Whitelist, then format. Never f-string a raw profile dict into a prompt."""
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
"""

        # temperature=0: this is structured extraction, not generation.
        raw = await self.ask_model(prompt, temperature=0, max_tokens=800)
        text = raw["text"] if isinstance(raw, dict) else raw

        try:
            return InterpretationResult.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as error:
            # Do NOT fabricate a result. Let the graph route this to CLARIFY.
            raise InterpretationError(f"invalid interpretation output: {error}") from error


def needs_clarification(result: InterpretationResult) -> bool:
    """Should the graph ask a question instead of translating?

    Provided here so orchestration has one place to call; the edge itself is
    the orchestration lead's to wire in.

    Ambiguity alone is deliberately NOT enough. Clarifying on every flagged
    ambiguity makes the system ask constantly and tanks Clarification Precision.
    Clarify only when the unresolved part is load-bearing for what the listener
    has to do.
    """
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
    """Trace fields for the run log. No chain-of-thought, no transcript text."""
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
    """Slice the outermost JSON object.

    Always slices, not only when fences are present -- a bare preamble sentence
    ("Here is the JSON:") would otherwise sail through and fail validation.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return text[start:end + 1]