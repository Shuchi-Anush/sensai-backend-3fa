"""
ScoringEngine — Strict integer scoring (0–10) for code evaluations.

Scoring rules:
  base_score = (passed_testcases / total_testcases) * 10

  - Compile-time error → score capped at 2
  - Runtime error (any) → score capped at 4
  - Partial success + errors → score capped at 4
  - Only fully correct → score can reach 10
"""

from dataclasses import dataclass
from api.engines.test_executor import TestCaseResult
from api.utils.logging import logger


@dataclass
class ScoringResult:
    score: int  # 0-10
    passed_testcases: int
    total_testcases: int
    status: str  # "SUCCESS" | "PARTIAL" | "FAILED"
    errors: list[str]


def compute_score(results: list[TestCaseResult]) -> ScoringResult:
    """
    Compute the final integer score (0-10) from test case results.

    Strict rules enforced:
    1. Compile error → score ≤ 2
    2. Runtime error → score ≤ 4
    3. Partial + any errors → score ≤ 4
    4. Fully correct → up to 10
    """
    total = len(results)
    if total == 0:
        return ScoringResult(
            score=0,
            passed_testcases=0,
            total_testcases=0,
            status="FAILED",
            errors=["No test cases to evaluate"],
        )

    passed = sum(1 for r in results if r.passed)
    base_score = round((passed / total) * 10)

    # Collect unique errors
    errors: list[str] = []
    has_compile_error = False
    has_runtime_error = False
    has_tle = False
    has_mle = False

    for r in results:
        if r.status == "COMPILE_ERROR":
            has_compile_error = True
            if r.error and r.error not in errors:
                errors.append(r.error)
        elif r.status == "RUNTIME_ERROR":
            has_runtime_error = True
            if r.error and r.error not in errors:
                errors.append(r.error)
        elif r.status == "TLE":
            has_tle = True
            if "Time Limit Exceeded" not in " ".join(errors):
                errors.append(f"Time Limit Exceeded on test case {r.test_case_id}")
        elif r.status == "MLE":
            has_mle = True
            if "Memory Limit Exceeded" not in " ".join(errors):
                errors.append(f"Memory Limit Exceeded on test case {r.test_case_id}")
        elif r.status == "WRONG_ANSWER":
            errors.append(
                f"Wrong answer on test case {r.test_case_id}: "
                f"expected '{r.expected_output[:50]}', got '{(r.actual_output or '')[:50]}'"
            )

    # Apply strict scoring caps
    final_score = base_score

    if has_compile_error:
        # Compile error → cap at 2
        final_score = min(final_score, 2)
    elif has_runtime_error or has_tle or has_mle:
        # Runtime/TLE/MLE → cap at 4
        final_score = min(final_score, 4)
    elif passed < total:
        # Partial success (wrong answers but no crashes) → cap at 4
        final_score = min(final_score, 4)

    # Determine overall status
    if passed == total:
        status = "SUCCESS"
    elif passed > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"

    logger.info(
        f"ScoringEngine — score={final_score}/10  passed={passed}/{total}  "
        f"status={status}  compile_err={has_compile_error}  runtime_err={has_runtime_error}  "
        f"tle={has_tle}  mle={has_mle}"
    )

    return ScoringResult(
        score=final_score,
        passed_testcases=passed,
        total_testcases=total,
        status=status,
        errors=errors,
    )
