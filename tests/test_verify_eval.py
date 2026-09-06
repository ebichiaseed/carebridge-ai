"""
Run the verify agent against verify_test_cases.json.

This is an evaluation harness, not a unit test: it calls the real model, so it
costs tokens and needs `aws sso login --profile hackathon` (or whichever
profile your team uses).

    python3 test_verify_eval.py
    python3 test_verify_eval.py --cases tests/verify_test_cases.json --only v002,v015
    python3 test_verify_eval.py --concurrency 2 -v

Exit code is 0 when every case's assert matches, 1 otherwise, so this can go
into CI later if you want -- same convention as test_interp_eval.py.

In addition to per-case PASS/FAIL, this also prints the verifier catch rate
and false-positive rate, since "did the assert match" and "did the verifier
do its job on a deliberately-wrong translation" are two different questions
worth tracking separately for this agent specifically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from agents.verify_agent import VerifyAgent
from configs.settings import VERIFICATION_MODEL
from models.model_factory import ModelFactory

DEFAULT_CASES = Path(__file__).resolve().parent / "verify_test_cases.json"
DEFAULT_RESULTS = Path("tests/verify_runs.jsonl")


# ---------------------------------------------------------------- checking

def check_case(case: dict, result) -> tuple[list[str], list[str]]:
    """Compare a result against the case's `assert` block.

    Returns (failures, skipped). A field named in `assert` that does not
    exist on VerificationResult is skipped rather than failed -- mirrors
    test_interp_eval.py's tolerance for schema drift ahead of the agent.
    """
    failures: list[str] = []
    skipped: list[str] = []

    for field, expected in case.get("assert", {}).items():
        if not hasattr(result, field):
            skipped.append(field)
            continue
        actual = getattr(result, field)
        if actual != expected:
            failures.append(f"{field}: expected {expected!r}, got {actual!r}")

    return failures, skipped


# ---------------------------------------------------------------- running

async def run_case(agent: VerifyAgent, case: dict, semaphore) -> dict:
    async with semaphore:
        started = time.perf_counter()

        result = await agent.run({
            "transcript": case["transcript"],
            "interpretation": case["interpretation"],
            "draft_translation": case["draft_translation"],
        })

        failures, skipped = check_case(case, result)

        record = {
            "id": case["id"],
            "probe": case.get("probe", ""),
            "category": case.get("category", ""),
            "is_injected_error": case.get("is_injected_error", False),
            "status": "PASS" if not failures else "FAIL",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "failures": failures,
            "skipped": skipped,
            "result": result.model_dump(),
            "watch_for": case.get("watch_for", ""),
        }
        return record


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--only", help="comma-separated case ids, e.g. v002,v015")
    parser.add_argument("--model", default=VERIFICATION_MODEL)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the full verification result for every case")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 1

    agent = VerifyAgent(model=ModelFactory.create(args.model))
    semaphore = asyncio.Semaphore(args.concurrency)

    records = await asyncio.gather(
        *(run_case(agent, case, semaphore) for case in cases)
    )
    records.sort(key=lambda r: r["id"])

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    mark = {"PASS": "ok  ", "FAIL": "FAIL"}
    all_skipped: set[str] = set()

    for record in records:
        print(f"{mark[record['status']]} {record['id']}  "
              f"{record['latency_ms']:>5} ms  {record['probe']}")
        for failure in record["failures"]:
            print(f"       - {failure}")
        if record["failures"] and record.get("watch_for"):
            print(f"       note: {record['watch_for']}")
        if args.verbose:
            print(f"       {json.dumps(record['result'], ensure_ascii=False)}")
        all_skipped.update(record["skipped"])

    passed = sum(r["status"] == "PASS" for r in records)

    # Verifier-specific metrics, on top of plain PASS/FAIL.
    injected = [r for r in records if r["is_injected_error"]]
    # "Should-pass" cases are genuine (non-injected) cases whose OWN assert
    # expects PASS. Genuine cases asserting CLARIFY (source ambiguity) are
    # deliberately not-PASS by design and must not count as false positives.
    case_by_id = {c["id"]: c for c in cases}
    should_pass = [
        r for r in records
        if not r["is_injected_error"]
        and case_by_id[r["id"]].get("assert", {}).get("verdict") == "PASS"
    ]
    caught = sum(1 for r in injected if r["result"]["verdict"] != "PASS")
    false_positives = sum(1 for r in should_pass if r["result"]["verdict"] != "PASS")

    print(f"\n{passed}/{len(records)} passed   "
          f"model={args.model}   results -> {args.results}")
    if injected:
        print(f"verifier catch rate: {caught}/{len(injected)} "
              f"({caught / len(injected):.0%})")
    if should_pass:
        print(f"false positive rate: {false_positives}/{len(should_pass)} "
              f"({false_positives / len(should_pass):.0%})")
    if all_skipped:
        print(f"fields asserted but missing from VerificationResult: "
              f"{', '.join(sorted(all_skipped))}")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
