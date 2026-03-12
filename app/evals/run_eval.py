"""Standalone evaluation script for the /api/chat endpoint.

Loads eval_dataset.json, calls the running backend's chat REST API for each
entry, and uses DeepEval's GEval (Correctness) metric with VLLMJudge to score
the responses.

Usage::

    python evals/run_eval.py

Requires env vars: OPENAI_API_ENDPOINT, OPENAI_API_TOKEN, OPENAI_MODEL.
Optionally set BACKEND_URL (default http://localhost:8888).
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from judge_model import VLLMJudge
from load_seed import load_seed

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8888").rstrip("/")
CHAT_ENDPOINT = f"{BACKEND_URL}/api/chat"
DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
THRESHOLD = 0.5


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def call_chat(question: str, description: str, session_id: str) -> str:
    payload = json.dumps(
        {
            "question": question,
            "description": description,
            "session_id": session_id,
        }
    ).encode()
    req = urllib.request.Request(
        CHAT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["answer"]


def run() -> None:
    dataset = load_dataset()
    judge = VLLMJudge()

    correctness = GEval(
        name="Correctness",
        criteria=(
            "The actual output must state the correct number and factual "
            "details matching the expected output. Minor wording differences "
            "are acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=THRESHOLD,
    )

    results: list[dict] = []

    for entry in dataset:
        entry_id = entry["id"]
        question = entry["question"]
        description = entry["description"]
        golden_answer = entry["golden_answer"]

        print(f"[{entry_id}] Calling chat endpoint ... ", end="", flush=True)
        try:
            actual_output = call_chat(question, description, session_id=entry_id)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append(
                {"id": entry_id, "score": 0.0, "passed": False, "error": str(exc)}
            )
            continue

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=golden_answer,
        )

        correctness.measure(test_case)
        passed = correctness.score >= THRESHOLD
        print(f"score={correctness.score:.2f}  {'PASS' if passed else 'FAIL'}")

        results.append(
            {
                "id": entry_id,
                "score": correctness.score,
                "passed": passed,
                "actual_output": actual_output,
            }
        )

    print_summary(results)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 72)
    print(f"{'ID':<50} {'SCORE':>6}  {'RESULT':>6}")
    print("-" * 72)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<50} {r['score']:>6.2f}  {tag:>6}")
    print("=" * 72)
    print(
        f"Total: {total}  Passed: {passed}  Failed: {failed}  "
        f"Pass rate: {passed / total * 100:.1f}%"
    )


if __name__ == "__main__":
    print("Seeding database ... ", end="", flush=True)
    counts = load_seed()
    print(
        f"done ({counts['persons']} persons, {counts['person_observations']} observations)"
    )
    run()
