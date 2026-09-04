# synthesiser.py

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.interpretation_agent import (
    InterpretationAgent,
    InterpretationError,
    needs_clarification,
)
from agents.structure_agent import StructureAgent, StructureError
from agents.verify_agent import VerifyAgent
from state import CareBridgeState


def _format_transcript(transcript: str | list[str]) -> str:
    """Convert multiple ASR candidates into text for the verifier."""

    if isinstance(transcript, str):
        return transcript

    return "\n".join(
        f"Candidate {index + 1}: {candidate}"
        for index, candidate in enumerate(transcript)
    )


def build_workflow(
    interpretation_agent: InterpretationAgent,
    structure_agent: StructureAgent,
    verify_agent: VerifyAgent,
):
    """
    Build and compile the text-only CareBridge workflow.

    Agents are passed in rather than created here so model configuration remains
    outside the graph and the workflow is easy to test with fake agents.
    """

    async def interpretation_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        try:
            result = await interpretation_agent.run(
                transcript=state["transcript"],
                glossary_hits=state.get("glossary_hits"),
                recent_context=state.get("recent_context"),
                person_info=state.get("person_info"),
            )

            return {
                "interpretation": result,
                "status": "processing",
                "error": None,
            }

        except InterpretationError as error:
            return {
                "status": "needs_clarification",
                "clarification_question": (
                    "Could you say that a different way?"
                ),
                "error": str(error),
            }

    async def structure_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        try:
            result = await structure_agent.run(
                interpretation=state["interpretation"],
                recent_context=state.get("recent_context"),
            )

            return {
                "draft_translation": result.draft_translation,
                "status": "processing",
                "error": None,
            }

        except StructureError as error:
            return {
                "status": "needs_clarification",
                "clarification_question": (
                    "Could you say that a different way?"
                ),
                "error": str(error),
            }

    async def verification_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        result = await verify_agent.run({
            "transcript": _format_transcript(state["transcript"]),
            "interpretation": state["interpretation"].model_dump(),
            "draft_translation": state["draft_translation"],
        })

        return {
            "verification": result,
            "verification_issues": result.issues,
        }

    def accept_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        """Expose text only after verification passes."""

        return {
            "final_text": state["draft_translation"],
            "status": "verified",
        }

    def clarify_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        """Select the most relevant clarification question."""

        question = state.get("clarification_question")

        verification = state.get("verification")
        if verification and verification.clarification_question:
            question = verification.clarification_question

        interpretation = state.get("interpretation")
        if (
            not question
            and interpretation
            and interpretation.clarification_question
        ):
            question = interpretation.clarification_question

        return {
            "clarification_question": (
                question or "Could you say that a different way?"
            ),
            "status": "needs_clarification",
        }

    def route_after_interpretation(
        state: CareBridgeState,
    ) -> str:
        if state["status"] == "needs_clarification":
            return "clarify"

        interpretation = state.get("interpretation")

        if interpretation is None:
            return "clarify"

        if needs_clarification(interpretation):
            return "clarify"

        return "structure"

    def route_after_structure(
        state: CareBridgeState,
    ) -> str:
        if state["status"] == "needs_clarification":
            return "clarify"

        if not state.get("draft_translation"):
            return "clarify"

        return "verify"

    def route_after_verification(
        state: CareBridgeState,
    ) -> str:
        verification = state.get("verification")

        if verification is None:
            return "clarify"

        if verification.verdict == "PASS":
            return "accept"

        # RETRY is routed to clarification for now because StructureAgent does
        # not yet accept verification feedback. Retrying the same deterministic
        # prompt would likely produce the same draft.
        return "clarify"

    builder = StateGraph(CareBridgeState)

    builder.add_node("interpret", interpretation_node)
    builder.add_node("structure", structure_node)
    builder.add_node("verify", verification_node)
    builder.add_node("accept", accept_node)
    builder.add_node("clarify", clarify_node)

    builder.add_edge(START, "interpret")

    builder.add_conditional_edges(
        "interpret",
        route_after_interpretation,
        {
            "structure": "structure",
            "clarify": "clarify",
        },
    )

    builder.add_conditional_edges(
        "structure",
        route_after_structure,
        {
            "verify": "verify",
            "clarify": "clarify",
        },
    )

    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "accept": "accept",
            "clarify": "clarify",
        },
    )

    builder.add_edge("accept", END)
    builder.add_edge("clarify", END)

    return builder.compile()