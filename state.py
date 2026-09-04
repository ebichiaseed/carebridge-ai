"""Shared state for the CareBridge agent workflow."""

from typing import Any, Literal, NotRequired, TypedDict

from agents.interpretation_agent import InterpretationResult
from agents.verify_agent import VerificationResult


class CareBridgeState(TypedDict):
    # Input
    transcript: str | list[str]

    # Context supplied to the interpretation agent
    glossary_hits: NotRequired[list[dict[str, Any]]]
    recent_context: NotRequired[list[dict[str, Any]]]
    person_info: NotRequired[dict[str, Any]]

    # Agent results
    interpretation: NotRequired[InterpretationResult]
    draft_translation: NotRequired[str]
    verification: NotRequired[VerificationResult]

    # Retry control
    retry_count: int
    max_retries: int
    verification_issues: NotRequired[list[str]]

    # Workflow result
    final_text: NotRequired[str]
    clarification_question: NotRequired[str]

    status: Literal[
        "processing",
        "verified",
        "needs_clarification",
        "failed",
    ]

    error: NotRequired[str]


def create_initial_state(
    transcript: str | list[str],
    *,
    glossary_hits: list[dict[str, Any]] | None = None,
    recent_context: list[dict[str, Any]] | None = None,
    person_info: dict[str, Any] | None = None,
    max_retries: int = 1,
) -> CareBridgeState:
    """Create a consistently initialized state for a new workflow run."""

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