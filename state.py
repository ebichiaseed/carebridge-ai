"""Shared LangGraph state for the CareBridge agent workflow."""

from typing import Any, Literal, NotRequired, TypedDict

from agents.interpretation_agent import InterpretationResult
from agents.structure_agent import StructureResult
from agents.verify_agent import VerificationResult


WorkflowStatus = Literal[
    "processing",
    "verified",
    "needs_clarification",
    "failed",
]


class CareBridgeState(TypedDict):
    """Information passed between CareBridge workflow nodes."""

    # Workflow input
    transcript: str | list[str]

    # Context for the interpretation agent
    glossary_hits: NotRequired[list[dict[str, Any]]]
    recent_context: NotRequired[list[dict[str, Any]]]
    person_info: NotRequired[dict[str, Any]]

    # Results produced by the three agents
    interpretation: NotRequired[InterpretationResult]
    structure: NotRequired[StructureResult]
    verification: NotRequired[VerificationResult]

    # Verification and retry information
    retry_count: int
    max_retries: int
    verification_issues: NotRequired[list[str]]

    # Terminal workflow output
    final_text: NotRequired[str]
    clarification_question: NotRequired[str]
    error: NotRequired[str]

    # Current workflow status
    status: WorkflowStatus


def create_initial_state(
    transcript: str | list[str],
    *,
    glossary_hits: list[dict[str, Any]] | None = None,
    recent_context: list[dict[str, Any]] | None = None,
    person_info: dict[str, Any] | None = None,
    max_retries: int = 1,
) -> CareBridgeState:
    """Create the initial state for one workflow execution."""

    if isinstance(transcript, str):
        if not transcript.strip():
            raise ValueError("transcript cannot be empty")
    elif not transcript or not any(item.strip() for item in transcript):
        raise ValueError("transcript candidates cannot be empty")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    return {
        "transcript": transcript,
        "glossary_hits": glossary_hits or [],
        "recent_context": recent_context or [],
        "person_info": person_info or {},
        "retry_count": 0,
        "max_retries": max_retries,
        "verification_issues": [],
        "status": "processing",
    }