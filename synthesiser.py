"""LangGraph orchestration for the CareBridge text workflow."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.interpretation_agent import (
    InterpretationAgent,
    InterpretationError,
    needs_clarification,
)
from agents.structure_agent import (
    StructureAgent,
    StructureError,
)
from agents.verify_agent import VerifyAgent
from state import CareBridgeState


def _format_transcript(transcript: str | list[str]) -> str:
    """Convert one or more ASR results into text for verification."""

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
    """Build and compile the CareBridge text workflow."""

    async def interpretation_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        """Run the interpretation agent."""

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
        """Create a natural-language draft from the interpretation."""

        try:
            result = await structure_agent.run(
                interpretation=state["interpretation"],
                recent_context=state.get("recent_context"),
            )

            return {
                "structure": result,
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
        """Verify that the draft preserves the original meaning."""

        result = await verify_agent.run({
            "transcript": _format_transcript(
                state["transcript"]
            ),
            "interpretation": (
                state["interpretation"].model_dump()
            ),
            "draft_translation": (
                state["structure"].draft_translation
            ),
        })

        return {
            "verification": result,
            "verification_issues": result.issues,
        }

    def accept_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        """Expose the draft as final text after verification passes."""

        return {
            "final_text": (
                state["structure"].draft_translation
            ),
            "status": "verified",
        }

    def clarify_node(
        state: CareBridgeState,
    ) -> dict[str, Any]:
        """Choose the most relevant clarification question."""

        question = state.get("clarification_question")

        verification = state.get("verification")

        if (
            verification is not None
            and verification.clarification_question
        ):
            question = verification.clarification_question

        interpretation = state.get("interpretation")

        if (
            not question
            and interpretation is not None
            and interpretation.clarification_question
        ):
            question = interpretation.clarification_question

        return {
            "clarification_question": (
                question
                or "Could you say that a different way?"
            ),
            "status": "needs_clarification",
        }

    def route_after_interpretation(
        state: CareBridgeState,
    ) -> str:
        """Decide whether to structure or request clarification."""

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
        """Decide whether the draft can be verified."""

        if state["status"] == "needs_clarification":
            return "clarify"

        structure = state.get("structure")

        if structure is None:
            return "clarify"

        if not structure.draft_translation.strip():
            return "clarify"

        return "verify"

    def route_after_verification(
        state: CareBridgeState,
    ) -> str:
        """Route according to the verifier verdict."""

        verification = state.get("verification")

        if verification is None:
            return "clarify"

        if verification.verdict == "PASS":
            return "accept"

        # RETRY and CLARIFY both stop for clarification for now.
        # A retry loop should only be added after StructureAgent can
        # receive the verifier's issues and revise its earlier draft.
        return "clarify"

    # Build the graph.
    builder = StateGraph(CareBridgeState)

    # Register nodes.
    builder.add_node(
        "interpret",
        interpretation_node,
    )
    builder.add_node(
        "structure",
        structure_node,
    )
    builder.add_node(
        "verify",
        verification_node,
    )
    builder.add_node(
        "accept",
        accept_node,
    )
    builder.add_node(
        "clarify",
        clarify_node,
    )

    # Starting edge.
    builder.add_edge(
        START,
        "interpret",
    )

    # Interpretation routing.
    builder.add_conditional_edges(
        "interpret",
        route_after_interpretation,
        {
            "structure": "structure",
            "clarify": "clarify",
        },
    )

    # Structure routing.
    builder.add_conditional_edges(
        "structure",
        route_after_structure,
        {
            "verify": "verify",
            "clarify": "clarify",
        },
    )

    # Verification routing.
    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "accept": "accept",
            "clarify": "clarify",
        },
    )

    # Terminal edges.
    builder.add_edge("accept", END)
    builder.add_edge("clarify", END)

    return builder.compile()