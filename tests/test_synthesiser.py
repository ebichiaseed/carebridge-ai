import json
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agents.interpretation_agent import InterpretationResult
from agents.interpretation_agent import InterpretationAgent
from agents.structure_agent import StructureAgent
from agents.structure_agent import StructureResult
from agents.verify_agent import VerifyAgent
from agents.verify_agent import VerificationResult
from configs.settings import (
    INTERPRETATION_MODEL,
    TRANSLATION_MODEL,
    VERIFICATION_MODEL,
)
from models.model_factory import ModelFactory
from state import create_initial_state
from synthesiser import build_workflow


class FakeInterpretationAgent:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return InterpretationResult(
            utterance_type="request",
            actor="caregiver",
            action="give",
            object="medicine",
            timing="after dinner",
        )


class FakeStructureAgent:
    def __init__(self):
        self.call_count = 0
        self.calls = []

    async def run(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        return StructureResult(draft_translation="Give the medicine after dinner.")


class SequencedVerifyAgent:
    def __init__(self, verdicts, fault_source="structure"):
        self.verdicts = iter(verdicts)
        self.fault_source = fault_source
        self.call_count = 0

    async def run(self, input_data):
        self.call_count += 1
        verdict = next(self.verdicts)
        return VerificationResult(
            verdict=verdict,
            issues=[] if verdict == "PASS" else ["timing needs review"],
            fault_source=None if verdict == "PASS" else self.fault_source,
        )


class SynthesiserRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_structure_retry_rebuilds_only_draft_with_feedback(self):
        interpretation = FakeInterpretationAgent()
        structure = FakeStructureAgent()
        verification = SequencedVerifyAgent(["RETRY", "PASS"])
        workflow = build_workflow(interpretation, structure, verification)

        result = await workflow.ainvoke(
            create_initial_state("Give the medicine after dinner", max_retries=1)
        )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(len(interpretation.calls), 1)
        self.assertEqual(structure.call_count, 2)
        self.assertEqual(verification.call_count, 2)
        self.assertEqual(
            structure.calls[1]["verification_issues"],
            ["timing needs review"],
        )

    async def test_interpretation_retry_rebuilds_full_pipeline(self):
        interpretation = FakeInterpretationAgent()
        structure = FakeStructureAgent()
        verification = SequencedVerifyAgent(
            ["RETRY", "PASS"], fault_source="interpretation"
        )
        workflow = build_workflow(interpretation, structure, verification)

        result = await workflow.ainvoke(
            create_initial_state("Give the medicine after dinner", max_retries=1)
        )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(len(interpretation.calls), 2)
        self.assertEqual(structure.call_count, 2)
        self.assertEqual(verification.call_count, 2)
        self.assertEqual(
            interpretation.calls[1]["verification_issues"],
            ["timing needs review"],
        )

    async def test_retry_stops_when_budget_is_exhausted(self):
        interpretation = FakeInterpretationAgent()
        structure = FakeStructureAgent()
        verification = SequencedVerifyAgent(["RETRY"])
        workflow = build_workflow(interpretation, structure, verification)

        result = await workflow.ainvoke(
            create_initial_state("Give the medicine", max_retries=0)
        )

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(len(interpretation.calls), 1)
        self.assertEqual(verification.call_count, 1)

    async def test_clarify_verdict_does_not_retry(self):
        interpretation = FakeInterpretationAgent()
        structure = FakeStructureAgent()
        verification = SequencedVerifyAgent(["CLARIFY"])
        workflow = build_workflow(interpretation, structure, verification)

        result = await workflow.ainvoke(
            create_initial_state("Give that one", max_retries=2)
        )

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(len(interpretation.calls), 1)


# Live evaluation -----------------------------------------------------------
# Opt in because this section uses real AWS credentials and incurs charges:
#   RUN_BEDROCK_EVAL=1 python3 -m unittest \
#       tests.test_synthesiser.BedrockSynthesiserEvaluation
# Results are overwritten as readable JSON after each run.

LIVE_RESULTS = Path(__file__).parent / "results" / "synthesiser_bedrock_eval.json"
LIVE_CASES_PATH = Path(__file__).parent / "synthesiser_test_cases.json"

# USD per million tokens for standard/on-demand inference. Override this whole
# mapping with BEDROCK_PRICE_PER_MILLION_JSON when account/region rates differ.
# The report records the exact rates used, so estimates remain auditable.
DEFAULT_PRICES = {
    "nova-lite": {"input": 0.06, "output": 0.24},
    "haiku": {"input": 1.00, "output": 5.00},
    "sonnet": {"input": 3.00, "output": 15.00},
}

QUALITY_THRESHOLDS = {
    "task_completion_rate": 0.75,
    "loop_discipline_rate": 1.0,
    "interpretation_accuracy_rate": 0.75,
    "semantic_equivalence_rate": 0.75,
    "schema_validation_pass_rate": 1.0,
}


class SemanticJudgeResult(BaseModel):
    meaning_equivalent: bool
    critical_facts_preserved: bool
    negation_preserved: bool
    timing_preserved: bool
    quantity_preserved: bool
    sequence_preserved: bool
    unsupported_information_added: bool
    reason: str


def _load_live_cases(path=LIVE_CASES_PATH):
    """Load and validate the repository's end-to-end synthesiser fixtures."""

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    required = {
        "id",
        "input",
        "expected_interpretation",
        "expected_final",
        "critical_facts",
        "expected_status",
        "safety_requirements",
    }
    adapted = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} in {path} must be a JSON object")
        missing = sorted(required - case.keys())
        if missing:
            raise ValueError(
                f"case {case.get('id', index)!r} is missing: {', '.join(missing)}"
            )
        if not isinstance(case["critical_facts"], list):
            raise ValueError(f"case {case['id']!r} critical_facts must be a list")
        if not isinstance(case["expected_interpretation"], dict):
            raise ValueError(
                f"case {case['id']!r} expected_interpretation must be an object"
            )
        if not isinstance(case["safety_requirements"], dict):
            raise ValueError(
                f"case {case['id']!r} safety_requirements must be an object"
            )
        if any(
            not isinstance(fact, str)
            and not (
                isinstance(fact, list)
                and fact
                and all(isinstance(option, str) for option in fact)
            )
            for fact in case["critical_facts"]
        ):
            raise ValueError(
                f"case {case['id']!r} critical_facts must contain strings "
                "or non-empty lists of strings"
            )

        adapted.append({**case, "transcript": case["input"]})
    return adapted


def _prices():
    override = os.getenv("BEDROCK_PRICE_PER_MILLION_JSON")
    return json.loads(override) if override else DEFAULT_PRICES


def _evaluation_configs():
    """Return mixed production config or homogeneous comparison configs."""

    mode = os.getenv("BEDROCK_EVAL_CONFIG", "comparison").strip().casefold()
    if mode in {"production", "mixed"}:
        return [{
            "name": os.getenv("BEDROCK_EVAL_CONFIG_NAME", "production"),
            "interpretation": os.getenv(
                "BEDROCK_EVAL_INTERPRETATION_MODEL", INTERPRETATION_MODEL
            ),
            "structure": os.getenv(
                "BEDROCK_EVAL_STRUCTURE_MODEL", TRANSLATION_MODEL
            ),
            "verification": os.getenv(
                "BEDROCK_EVAL_VERIFICATION_MODEL", VERIFICATION_MODEL
            ),
        }]
    if mode != "comparison":
        raise ValueError(
            "BEDROCK_EVAL_CONFIG must be 'comparison', 'production', or 'mixed'"
        )

    names = [
        item.strip() for item in os.getenv(
            "BEDROCK_EVAL_MODELS", "nova-lite,haiku,sonnet"
        ).split(",") if item.strip()
    ]
    if not names:
        raise ValueError("BEDROCK_EVAL_MODELS did not select any models")
    return [
        {
            "name": name,
            "interpretation": name,
            "structure": name,
            "verification": name,
        }
        for name in names
    ]


def _usage_since(model, start):
    usage = model.usage_history[start:]
    input_tokens = sum(item.get("inputTokens", 0) for item in usage)
    output_tokens = sum(item.get("outputTokens", 0) for item in usage)
    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }


def _estimated_cost(usage, model_name, prices):
    rate = prices.get(model_name)
    if rate is None:
        return None
    return round(
        (
            usage["input"] * rate["input"]
            + usage["output"] * rate["output"]
        ) / 1_000_000,
        8,
    )


def _schema_checks(result):
    checks = []
    interpretation = result.get("interpretation")
    if interpretation is not None:
        checks.append(isinstance(interpretation, InterpretationResult))
    structure = result.get("structure")
    if structure is not None:
        checks.append(isinstance(structure, StructureResult))
    verification = result.get("verification")
    if verification is not None:
        parsed = isinstance(verification, VerificationResult)
        parse_fallback = any(
            "unparseable response" in issue for issue in verification.issues
        )
        checks.append(parsed and not parse_fallback)
    return checks


def _check_interpretation(expected, actual):
    """Deterministically check labelled InterpretationResult fields."""

    checks = {}
    if actual is None:
        return {key: False for key in expected if key != "semantic_summary"}

    for field, wanted in expected.items():
        if field == "semantic_summary":
            continue
        got = getattr(actual, field, None)
        if isinstance(wanted, list):
            checks[field] = got in wanted
        else:
            checks[field] = got == wanted
    return checks


async def _judge_semantics(judge_model, case, result, answer):
    """Use a separately configured model to judge flexible semantic equivalence."""

    interpretation = result.get("interpretation")
    verification = result.get("verification")
    payload = {
        "original_transcript": case["transcript"],
        "expected_interpretation": case["expected_interpretation"],
        "expected_final": case["expected_final"],
        "critical_facts": case["critical_facts"],
        "safety_requirements": case["safety_requirements"],
        "actual_status": result.get("status"),
        "actual_interpretation": (
            interpretation.model_dump() if interpretation is not None else None
        ),
        "actual_final": answer,
        "verifier": (
            verification.model_dump() if verification is not None else None
        ),
    }
    prompt = """You are an independent evaluator for a caregiving translation
workflow. Compare the actual result with the labelled reference. Accept natural
paraphrases, but be strict about negation, timing, quantities, ordering, people,
objects, and invented details. Judge the labelled meaning, not word overlap.

Return only this JSON shape:
{"meaning_equivalent": boolean, "critical_facts_preserved": boolean,
"negation_preserved": boolean, "timing_preserved": boolean,
"quantity_preserved": boolean, "sequence_preserved": boolean,
"unsupported_information_added": boolean, "reason": string}

CASE:
""" + json.dumps(payload, ensure_ascii=False)
    raw = await judge_model.generate(prompt=prompt, temperature=0, max_tokens=500)
    text = raw["text"] if isinstance(raw, dict) else raw
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return SemanticJudgeResult.model_validate_json(text[start:end])
    except (ValueError, ValidationError) as error:
        raise ValueError(f"invalid semantic judge output: {error}") from error


def _semantic_pass(judgement, safety):
    required = [
        judgement.meaning_equivalent,
        judgement.critical_facts_preserved,
        not judgement.unsupported_information_added,
    ]
    if safety.get("must_preserve_negation"):
        required.append(judgement.negation_preserved)
    if safety.get("must_preserve_quantity"):
        required.append(judgement.quantity_preserved)
    if safety.get("must_preserve_sequence"):
        required.append(judgement.sequence_preserved)
    return all(required)


def _applicable_rate(cases, requirement, judgement_field):
    applicable = [
        item for item in cases
        if item.get("safety_requirements", {}).get(requirement)
    ]
    if not applicable:
        return None
    return sum(
        bool(item.get("semantic_judgement", {}).get(judgement_field))
        for item in applicable
    ) / len(applicable)


@unittest.skipUnless(
    os.getenv("RUN_BEDROCK_EVAL") == "1",
    "set RUN_BEDROCK_EVAL=1 to call paid AWS Bedrock models",
)
class BedrockSynthesiserEvaluation(unittest.IsolatedAsyncioTestCase):
    """End-to-end, paid evaluation against each configured Bedrock model."""

    async def test_live_models_and_write_human_readable_report(self):
        from tools.glossary_lookup import get_retriever

        live_cases = _load_live_cases()
        glossary = get_retriever()
        judge_name = os.getenv("BEDROCK_EVAL_JUDGE_MODEL", "sonnet")
        judge_model = ModelFactory.create(judge_name)
        evaluation_configs = _evaluation_configs()
        prices = _prices()
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Live end-to-end AWS Bedrock synthesiser evaluation",
            "cases_path": str(LIVE_CASES_PATH),
            "case_count": len(live_cases),
            "semantic_judge_model": judge_name,
            "evaluation_mode": os.getenv(
                "BEDROCK_EVAL_CONFIG", "comparison"
            ),
            "quality_thresholds": QUALITY_THRESHOLDS,
            "thresholds_enforced": (
                os.getenv("BEDROCK_EVAL_ENFORCE_THRESHOLDS") == "1"
            ),
            "metric_definitions": {
                "token_cost": "Bedrock-reported tokens and estimated USD using recorded per-million-token rates.",
                "loop_discipline": "Run terminates without exceeding its configured retry budget.",
                "task_completion_rate": "Cases ending in their expected terminal status.",
                "interpretation_accuracy": "Exact checks of labelled categorical fields in the structured interpretation.",
                "semantic_equivalence": "Independent model judgement against expected meaning and safety requirements.",
                "verifier_false_pass": "Final verifier returned PASS although the independent semantic judgement failed.",
                "schema_validation_pass_rate": "Produced agent artifacts accepted by their Pydantic schema; verifier fallbacks fail.",
            },
            "models": [],
        }

        failures = []
        LIVE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        try:
            for config in evaluation_configs:
                stage_models = {
                    stage: ModelFactory.create(model_name)
                    for stage, model_name in config.items()
                    if stage != "name"
                }
                workflow = build_workflow(
                    InterpretationAgent(stage_models["interpretation"]),
                    StructureAgent(stage_models["structure"]),
                    VerifyAgent(stage_models["verification"]),
                )
                model_record = {
                    "name": config["name"],
                    "configuration": {
                        stage: model_name
                        for stage, model_name in config.items()
                        if stage != "name"
                    },
                    "judge_is_same_model_family": (
                        judge_name in {
                            model_name
                            for stage, model_name in config.items()
                            if stage != "name"
                        }
                    ),
                    "cases": [],
                }

                for case in live_cases:
                    started = time.perf_counter()
                    usage_starts = {
                        stage: len(model.usage_history)
                        for stage, model in stage_models.items()
                    }
                    judge_usage_start = len(judge_model.usage_history)
                    try:
                        glossary_result = await glossary.alookup(case["transcript"])
                        result = await workflow.ainvoke(create_initial_state(
                            case["transcript"],
                            glossary_hits=glossary_result.hits,
                            recent_context=case.get("recent_context"),
                            person_info=case.get("person_info"),
                            max_retries=1,
                        ))
                        schema_checks = _schema_checks(result)
                        completed = result["status"] == case["expected_status"]
                        bounded = result["retry_count"] <= result["max_retries"]
                        loop_disciplined = bounded
                        answer = (
                            result.get("final_text")
                            or result.get("clarification_question")
                            or ""
                        )
                        interpretation_checks = _check_interpretation(
                            case["expected_interpretation"],
                            result.get("interpretation"),
                        )
                        judgement = await _judge_semantics(
                            judge_model, case, result, answer
                        )
                        semantic_pass = completed and _semantic_pass(
                            judgement, case["safety_requirements"]
                        )
                        verification = result.get("verification")
                        verifier_false_pass = bool(
                            verification is not None
                            and verification.verdict == "PASS"
                            and not semantic_pass
                        )
                        record = {
                            "id": case["id"],
                            "transcript": case["transcript"],
                            "languages": case.get("languages", []),
                            "expected_interpretation": case["expected_interpretation"],
                            "expected_final": case["expected_final"],
                            "critical_facts": case["critical_facts"],
                            "safety_requirements": case["safety_requirements"],
                            "transformation_required": case.get(
                                "transformation_required"
                            ),
                            "expected_status": case["expected_status"],
                            "actual_status": result["status"],
                            "answer": answer,
                            "interpretation": (
                                result["interpretation"].model_dump()
                                if result.get("interpretation") is not None
                                else None
                            ),
                            "retry_count": result["retry_count"],
                            "max_retries": result["max_retries"],
                            "loop_disciplined": loop_disciplined,
                            "task_completed": completed,
                            "interpretation_checks": interpretation_checks,
                            "interpretation_pass": (
                                bool(interpretation_checks)
                                and all(interpretation_checks.values())
                            ),
                            "semantic_judgement": judgement.model_dump(),
                            "semantic_pass": semantic_pass,
                            "verifier_false_pass": verifier_false_pass,
                            "repair_attempted": result["retry_count"] > 0,
                            "repair_succeeded": (
                                result["retry_count"] > 0 and semantic_pass
                            ),
                            "schema_checks": len(schema_checks),
                            "schema_passes": sum(schema_checks),
                            "glossary": {
                                **glossary_result.to_trace(),
                                "hits": glossary_result.hits,
                            },
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                        }
                    except Exception as error:  # keep the multi-model report useful
                        record = {
                            "id": case["id"],
                            "transcript": case["transcript"],
                            "safety_requirements": case["safety_requirements"],
                            "error": f"{type(error).__name__}: {error}",
                            "loop_disciplined": False,
                            "task_completed": False,
                            "interpretation_pass": False,
                            "semantic_pass": False,
                            "verifier_false_pass": False,
                            "repair_attempted": False,
                            "repair_succeeded": False,
                            "schema_checks": 1,
                            "schema_passes": 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                        }
                        failures.append(
                            f"{config['name']}/{case['id']}: {record['error']}"
                        )

                    stage_usage = {
                        stage: _usage_since(model, usage_starts[stage])
                        for stage, model in stage_models.items()
                    }
                    input_tokens = sum(
                        usage["input"] for usage in stage_usage.values()
                    )
                    output_tokens = sum(
                        usage["output"] for usage in stage_usage.values()
                    )
                    record["token_usage"] = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                        "by_stage": stage_usage,
                    }
                    stage_costs = {
                        stage: _estimated_cost(
                            stage_usage[stage], config[stage], prices
                        )
                        for stage in stage_models
                    }
                    record["estimated_cost_usd"] = round(
                        sum(cost or 0 for cost in stage_costs.values()), 8
                    )
                    record["estimated_cost_by_stage_usd"] = stage_costs
                    record["judge_token_usage"] = _usage_since(
                        judge_model, judge_usage_start
                    )
                    record["estimated_judge_cost_usd"] = _estimated_cost(
                        record["judge_token_usage"], judge_name, prices
                    )
                    model_record["cases"].append(record)

                cases = model_record["cases"]
                schema_total = sum(item["schema_checks"] for item in cases)
                model_record["summary"] = {
                    "task_completion_rate": sum(item["task_completed"] for item in cases) / len(cases),
                    "loop_discipline_rate": sum(item["loop_disciplined"] for item in cases) / len(cases),
                    "interpretation_accuracy_rate": sum(item["interpretation_pass"] for item in cases) / len(cases),
                    "semantic_equivalence_rate": sum(item["semantic_pass"] for item in cases) / len(cases),
                    "verifier_false_pass_rate": sum(item["verifier_false_pass"] for item in cases) / len(cases),
                    "repair_attempts": sum(item["repair_attempted"] for item in cases),
                    "successful_repairs": sum(item["repair_succeeded"] for item in cases),
                    "negation_preservation_rate": _applicable_rate(
                        cases, "must_preserve_negation", "negation_preserved"
                    ),
                    "sequence_preservation_rate": _applicable_rate(
                        cases, "must_preserve_sequence", "sequence_preserved"
                    ),
                    "quantity_preservation_rate": _applicable_rate(
                        cases, "must_preserve_quantity", "quantity_preserved"
                    ),
                    "unsupported_information_rate": sum(
                        item.get("semantic_judgement", {}).get(
                            "unsupported_information_added", False
                        )
                        for item in cases
                    ) / len(cases),
                    "schema_validation_pass_rate": sum(item["schema_passes"] for item in cases) / schema_total,
                    "input_tokens": sum(item["token_usage"]["input"] for item in cases),
                    "output_tokens": sum(item["token_usage"]["output"] for item in cases),
                    "estimated_cost_usd": round(sum(item["estimated_cost_usd"] or 0 for item in cases), 8),
                    "estimated_judge_cost_usd": round(
                        sum(item["estimated_judge_cost_usd"] or 0 for item in cases),
                        8,
                    ),
                    "token_usage_by_stage": {
                        stage: {
                            token_type: sum(
                                item["token_usage"]["by_stage"][stage][token_type]
                                for item in cases
                            )
                            for token_type in ("input", "output", "total")
                        }
                        for stage in stage_models
                    },
                    "estimated_cost_by_stage_usd": {
                        stage: round(
                            sum(
                                item["estimated_cost_by_stage_usd"][stage] or 0
                                for item in cases
                            ),
                            8,
                        )
                        for stage in stage_models
                    },
                    "price_per_million_tokens_usd": {
                        stage: prices.get(config[stage])
                        for stage in stage_models
                    },
                }
                model_record["summary"]["meets_quality_thresholds"] = all(
                    model_record["summary"][metric] >= threshold
                    for metric, threshold in QUALITY_THRESHOLDS.items()
                )
                report["models"].append(model_record)
        finally:
            LIVE_RESULTS.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        self.assertFalse(failures, "\n".join(failures))
        if report["thresholds_enforced"]:
            for model_record in report["models"]:
                with self.subTest(configuration=model_record["name"]):
                    self.assertTrue(
                        model_record["summary"]["meets_quality_thresholds"],
                        model_record["summary"],
                    )


if __name__ == "__main__":
    unittest.main()
