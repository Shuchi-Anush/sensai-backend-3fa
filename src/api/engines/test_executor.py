"""
TestCaseExecutor — Runs source code against a list of test cases.

For each test case, calls CodeRunner with stdin = test_case.input,
then compares stdout with test_case.expected_output.
"""

from dataclasses import dataclass
from api.engines.code_runner import run_python_code, RunStatus
from api.utils.logging import logger


@dataclass
class TestCaseResult:
    test_case_id: int
    input: str
    expected_output: str
    actual_output: str | None = None
    passed: bool = False
    status: str = "PENDING"  # PASSED | WRONG_ANSWER | RUNTIME_ERROR | TLE | MLE | COMPILE_ERROR
    error: str | None = None
    execution_time_ms: float = 0.0


async def execute_test_cases(
    source_code: str,
    language: str,
    test_cases: list[dict],
    timeout_per_case: float = 5.0,
) -> list[TestCaseResult]:
    """
    Execute source_code against each test case and return results.

    Each test_case dict should have:
      - id: int
      - input: str
      - expected_output: str
    """
    results: list[TestCaseResult] = []

    if language != "python":
        # For now, only Python is supported
        for tc in test_cases:
            results.append(TestCaseResult(
                test_case_id=tc["id"],
                input=tc["input"],
                expected_output=tc["expected_output"],
                status="UNSUPPORTED_LANGUAGE",
                error=f"Language '{language}' is not yet supported. Only Python is available.",
            ))
        return results

    for tc in test_cases:
        tc_result = TestCaseResult(
            test_case_id=tc["id"],
            input=tc["input"],
            expected_output=tc["expected_output"],
        )

        try:
            run_result = await run_python_code(
                source_code=source_code,
                stdin_data=tc["input"],
                timeout_seconds=timeout_per_case,
            )

            tc_result.execution_time_ms = run_result.execution_time_ms

            if run_result.status == RunStatus.COMPILE_ERROR:
                tc_result.status = "COMPILE_ERROR"
                tc_result.error = run_result.stderr
                tc_result.passed = False
            elif run_result.status == RunStatus.TLE:
                tc_result.status = "TLE"
                tc_result.error = run_result.stderr
                tc_result.passed = False
            elif run_result.status == RunStatus.MLE:
                tc_result.status = "MLE"
                tc_result.error = run_result.stderr
                tc_result.passed = False
            elif run_result.status == RunStatus.RUNTIME_ERROR:
                tc_result.status = "RUNTIME_ERROR"
                tc_result.error = run_result.stderr
                tc_result.actual_output = run_result.stdout
                tc_result.passed = False
            else:
                # Compare outputs (strip whitespace for leniency)
                actual = (run_result.stdout or "").strip()
                expected = tc["expected_output"].strip()
                tc_result.actual_output = actual

                if actual == expected:
                    tc_result.status = "PASSED"
                    tc_result.passed = True
                else:
                    tc_result.status = "WRONG_ANSWER"
                    tc_result.passed = False

        except Exception as e:
            logger.error(f"TestCaseExecutor error on test {tc['id']}: {e}")
            tc_result.status = "RUNTIME_ERROR"
            tc_result.error = str(e)
            tc_result.passed = False

        results.append(tc_result)

        # If compile error on first test case, skip the rest (same code won't compile again)
        if tc_result.status == "COMPILE_ERROR":
            for remaining_tc in test_cases[test_cases.index(tc) + 1:]:
                results.append(TestCaseResult(
                    test_case_id=remaining_tc["id"],
                    input=remaining_tc["input"],
                    expected_output=remaining_tc["expected_output"],
                    status="COMPILE_ERROR",
                    error=tc_result.error,
                    passed=False,
                ))
            break

    logger.info(
        f"TestCaseExecutor — {sum(1 for r in results if r.passed)}/{len(results)} passed"
    )
    return results
