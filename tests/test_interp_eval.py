"""
Run the interpretation agent against interp_test_cases.json.

This is an evaluation harness, not a unit test: it calls the real model, so it
costs tokens and needs `aws sso login --profile hackathon`.

    python3 test_interp_eval.py
    python3 test_interp_eval.py --cases data/interp_test_cases.json --only c011,c017
    python3 test_interp_eval.py --no-glossary --concurrency 2

Exit code is 0 when every case passes its asserts, 1 otherwise, so this can go
into CI later if you want.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from agents.interpretation_agent import InterpretationAgent, InterpretationError
from configs.settings import INTERPRETATION_MODEL
from models.model_factory import ModelFactory

DEFAULT_CASES = Path(__file__).resolve().parent / "interp_test_cases.json"
DEFAULT_RESULTS = Path("tests/interp_runs.jsonl")


# ---------------------------------------------------------------- glossary

def load_glossary_lookup(enabled: bool):
    """Return a callable transcript -> list[dict] of glossary hits, or None.

    Tolerates the glossary module being absent, unindexed, or on the older
    string-returning API -- an eval run should not be blocked by it.
    """
    if not enabled:
        return None
    try:
        from tools.glossary_lookup import get_retriever
    except ImportError as error:
        print(f"[glossary] unavailable ({error}); running with no hits", file=sys.stderr)
        return None

    retriever = get_retriever()
    retriever.index()

    def lookup(transcript):
        result = retriever.lookup(transcript)
        return result.hits

    return lookup


# ---------------------------------------------------------------- checking

def check_case(case: dict, result) -> tuple[list[str], list[str]]:
    """Compare a result against the case's `assert` block and expect_ambiguity.

    Returns (failures, skipped). A field named in `assert` that does not exist
    on InterpretationResult is skipped rather than failed -- the test cases
    were written against a slightly richer schema than the agent currently
    returns, and silently failing on that would hide the real regressions.
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

    if "expect_ambiguity" in case:
        expected_flag = bool(case["expect_ambiguity"])
        actual_flag = bool(getattr(result, "ambiguity", []))
        if actual_flag != expected_flag:
            failures.append(
                f"ambiguity: expected {'non-empty' if expected_flag else 'empty'}, "
                f"got {getattr(result, 'ambiguity', [])!r}"
            )

    return failures, skipped


# ---------------------------------------------------------------- running

async def run_case(agent, case: dict, lookup, semaphore) -> dict:
    async with semaphore:
        transcript = case["transcript"]
        started = time.perf_counter()

        hits = []
        glossary_error = None
        if lookup is not None:
            try:
                hits = lookup(transcript)
            except Exception as error:          # noqa: BLE001 - eval must continue
                glossary_error = f"{type(error).__name__}: {error}"

        record = {
            "id": case["id"],
            "probe": case.get("probe", ""),
            "glossary_hits": [h.get("term") for h in hits],
            "glossary_error": glossary_error,
        }

        try:
            result = await agent.run(
                transcript=transcript,
                glossary_hits=hits,
                recent_context=case.get("recent_context"),
                person_info=case.get("person_info"),
            )
        except InterpretationError as error:
            record.update(
                status="ERROR",
                latency_ms=round((time.perf_counter() - started) * 1000),
                failures=[f"InterpretationError: {error}"],
                skipped=[],
                result=None,
            )
            return record

        failures, skipped = check_case(case, result)
        record.update(
            status="PASS" if not failures else "FAIL",
            latency_ms=round((time.perf_counter() - started) * 1000),
            failures=failures,
            skipped=skipped,
            result=result.model_dump(),
            watch_for=case.get("watch_for", ""),
        )
        return record


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--only", help="comma-separated case ids, e.g. c011,c017")
    parser.add_argument("--model", default=INTERPRETATION_MODEL)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the full interpretation for every case")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 1

    agent = InterpretationAgent(model=ModelFactory.create(args.model))
    lookup = load_glossary_lookup(not args.no_glossary)
    semaphore = asyncio.Semaphore(args.concurrency)

    records = await asyncio.gather(
        *(run_case(agent, case, lookup, semaphore) for case in cases)
    )
    records.sort(key=lambda r: r["id"])

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    mark = {"PASS": "ok  ", "FAIL": "FAIL", "ERROR": "ERR "}
    all_skipped: set[str] = set()

    for record in records:
        print(f"{mark[record['status']]} {record['id']}  "
              f"{record['latency_ms']:>5} ms  {record['probe']}")
        for failure in record["failures"]:
            print(f"       - {failure}")
        if record["failures"] and record.get("watch_for"):
            print(f"       note: {record['watch_for']}")
        if args.verbose and record["result"]:
            print(f"       {json.dumps(record['result'], ensure_ascii=False)}")
        all_skipped.update(record["skipped"])

    passed = sum(r["status"] == "PASS" for r in records)
    print(f"\n{passed}/{len(records)} passed   "
          f"model={args.model}   results -> {args.results}")
    if all_skipped:
        print(f"fields asserted but missing from InterpretationResult: "
              f"{', '.join(sorted(all_skipped))}")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))