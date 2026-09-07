import asyncio

from agents.interpretation_agent import InterpretationAgent
from agents.structure_agent import StructureAgent
from agents.verify_agent import VerifyAgent
from models.model_factory import ModelFactory
from synthesiser import build_workflow


async def main():
    model = ModelFactory.create("haiku")

    workflow = build_workflow(
        interpretation_agent=InterpretationAgent(model=model),
        structure_agent=StructureAgent(model=model),
        verify_agent=VerifyAgent(model=model),
    )

    initial_state = {
        "transcript": "Give ah ma her medicine after dinner.",
        "glossary_hits": [],
        "recent_context": [],
        "person_info": {},
        "retry_count": 0,
        "max_retries": 1,
        "status": "processing",
    }

    result = await workflow.ainvoke(initial_state)

    print(result)

    if result["status"] == "verified":
        print("Verified text:", result["final_text"])
    else:
        print(
            "Clarification needed:",
            result["clarification_question"],
        )


if __name__ == "__main__":
    asyncio.run(main())