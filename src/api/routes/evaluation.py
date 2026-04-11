import asyncio
import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from pydantic import BaseModel, Field
from api.config import openai_plan_to_model_name
from api.models import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationScores,
    EvaluationWeights,
    EvaluationFeedback,
    EvaluatorFeedback,
    CodeEvaluationRequest,
    CodeEvaluationResponse,
    CodeTestCaseResult,
)
from api.llm import run_llm_with_openai
from api.prompts import compile_prompt
from api.prompts.evaluation import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    AUTO_SCORE_SYSTEM_PROMPT,
    AUTO_SCORE_USER_PROMPT,
    AI_SCORE_SYSTEM_PROMPT,
    AI_SCORE_USER_PROMPT,
    HUMAN_SCORE_SYSTEM_PROMPT,
    HUMAN_SCORE_USER_PROMPT,
)
from api.db.evaluation import create_evaluation, get_evaluations, get_evaluation_by_id
from api.db.code_evaluation import create_code_evaluation, get_test_cases_for_question
from api.db.code_draft import get_user_code_draft
from api.engines.test_executor import execute_test_cases
from api.engines.scoring_engine import compute_score
from api.utils.logging import logger

router = APIRouter()

# ---------- Fixed Questions for Code Autograding ----------

FIXED_QUESTIONS: dict[str, dict] = {
    "binary_search": {
        "id": "binary_search",
        "title": "Binary Search",
        "description": (
            "Given a sorted integer array and a target value, find the 0-based index "
            "of the target using binary search. Print -1 if the target is not found."
        ),
        "input_format": "Line 1: space-separated sorted integers\nLine 2: target integer",
        "output_format": "Index of target (0-based), or -1 if not found",
        "templates": {
            "python": (
                "arr = list(map(int, input().split()))\n"
                "target = int(input())\n\n"
                "def binary_search(arr, target):\n"
                "    # Write your implementation here\n"
                "    pass\n\n"
                "print(binary_search(arr, target))"
            ),
            "javascript": (
                "const lines = require('fs').readFileSync(0, 'utf8').trim().split('\\n');\n"
                "const arr = lines[0].trim().split(' ').map(Number);\n"
                "const target = Number(lines[1].trim());\n\n"
                "function binarySearch(arr, target) {\n"
                "    // Write your implementation here\n"
                "}\n\n"
                "console.log(binarySearch(arr, target));"
            ),
            "java": (
                "import java.util.Scanner;\n\n"
                "public class Solution {\n"
                "    static int binarySearch(int[] arr, int target) {\n"
                "        // Write your implementation here\n"
                "        return -1;\n"
                "    }\n\n"
                "    public static void main(String[] args) {\n"
                "        Scanner sc = new Scanner(System.in);\n"
                "        String[] parts = sc.nextLine().trim().split(\"\\\\s+\");\n"
                "        int[] arr = new int[parts.length];\n"
                "        for (int i = 0; i < parts.length; i++) arr[i] = Integer.parseInt(parts[i]);\n"
                "        int target = Integer.parseInt(sc.nextLine().trim());\n"
                "        System.out.println(binarySearch(arr, target));\n"
                "    }\n"
                "}"
            ),
            "cpp": (
                "#include <iostream>\n"
                "#include <vector>\n"
                "#include <sstream>\n"
                "using namespace std;\n\n"
                "int binarySearch(vector<int>& arr, int target) {\n"
                "    // Write your implementation here\n"
                "    return -1;\n"
                "}\n\n"
                "int main() {\n"
                "    string line;\n"
                "    getline(cin, line);\n"
                "    istringstream iss(line);\n"
                "    vector<int> arr;\n"
                "    int x;\n"
                "    while (iss >> x) arr.push_back(x);\n"
                "    int target;\n"
                "    cin >> target;\n"
                "    cout << binarySearch(arr, target) << endl;\n"
                "    return 0;\n"
                "}"
            ),
        },
        "test_cases": [
            {"id": 1, "input": "1 3 5 7 9\n5", "expected_output": "2"},
            {"id": 2, "input": "2 4 6 8 10\n1", "expected_output": "-1"},
            {"id": 3, "input": "1\n1", "expected_output": "0"},
            {"id": 4, "input": "1 2 3 4 5\n5", "expected_output": "4"},
            {"id": 5, "input": "10 20 30 40 50\n30", "expected_output": "2"},
        ],
    },
    "fibonacci": {
        "id": "fibonacci",
        "title": "Fibonacci Series",
        "description": (
            "Given N, print the first N Fibonacci numbers (F(0)=0, F(1)=1, F(2)=1, ...) "
            "separated by spaces on a single line."
        ),
        "input_format": "Single integer N (1 ≤ N ≤ 50)",
        "output_format": "First N Fibonacci numbers separated by spaces",
        "templates": {
            "python": (
                "n = int(input())\n\n"
                "# Print first n Fibonacci numbers separated by spaces\n"
                "# F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...\n"
            ),
            "javascript": (
                "const n = parseInt(require('fs').readFileSync(0, 'utf8').trim());\n\n"
                "// Print first n Fibonacci numbers separated by spaces\n"
                "// F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...\n"
            ),
            "java": (
                "import java.util.Scanner;\n\n"
                "public class Solution {\n"
                "    public static void main(String[] args) {\n"
                "        Scanner sc = new Scanner(System.in);\n"
                "        int n = Integer.parseInt(sc.nextLine().trim());\n"
                "        // Print first n Fibonacci numbers separated by spaces\n"
                "        // F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...\n"
                "    }\n"
                "}"
            ),
            "cpp": (
                "#include <iostream>\n"
                "using namespace std;\n\n"
                "int main() {\n"
                "    int n;\n"
                "    cin >> n;\n"
                "    // Print first n Fibonacci numbers separated by spaces\n"
                "    // F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...\n"
                "    return 0;\n"
                "}"
            ),
        },
        "test_cases": [
            {"id": 1, "input": "1", "expected_output": "0"},
            {"id": 2, "input": "5", "expected_output": "0 1 1 2 3"},
            {"id": 3, "input": "8", "expected_output": "0 1 1 2 3 5 8 13"},
            {"id": 4, "input": "2", "expected_output": "0 1"},
            {"id": 5, "input": "10", "expected_output": "0 1 1 2 3 5 8 13 21 34"},
        ],
    },
    "palindrome": {
        "id": "palindrome",
        "title": "Reverse Palindrome",
        "description": (
            "Given a string, check if it reads the same forwards and backwards "
            "(i.e., is a palindrome). Print 'Yes' if it is, 'No' otherwise."
        ),
        "input_format": "A single string (no spaces)",
        "output_format": "'Yes' if palindrome, 'No' otherwise",
        "templates": {
            "python": (
                "s = input()\n\n"
                "# Check if s is a palindrome\n"
                "# Print 'Yes' or 'No'\n"
            ),
            "javascript": (
                "const s = require('fs').readFileSync(0, 'utf8').trim();\n\n"
                "// Check if s is a palindrome\n"
                "// Print 'Yes' or 'No'\n"
            ),
            "java": (
                "import java.util.Scanner;\n\n"
                "public class Solution {\n"
                "    public static void main(String[] args) {\n"
                "        Scanner sc = new Scanner(System.in);\n"
                "        String s = sc.nextLine().trim();\n"
                "        // Check if s is a palindrome\n"
                "        // Print 'Yes' or 'No'\n"
                "    }\n"
                "}"
            ),
            "cpp": (
                "#include <iostream>\n"
                "#include <string>\n"
                "#include <algorithm>\n"
                "using namespace std;\n\n"
                "int main() {\n"
                "    string s;\n"
                "    cin >> s;\n"
                "    // Check if s is a palindrome\n"
                "    // Print 'Yes' or 'No'\n"
                "    return 0;\n"
                "}"
            ),
        },
        "test_cases": [
            {"id": 1, "input": "racecar", "expected_output": "Yes"},
            {"id": 2, "input": "hello", "expected_output": "No"},
            {"id": 3, "input": "madam", "expected_output": "Yes"},
            {"id": 4, "input": "abcba", "expected_output": "Yes"},
            {"id": 5, "input": "python", "expected_output": "No"},
        ],
    },
}


# ---------- LLM Response Models ----------


class ClassificationOutput(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the classification")
    input_type: str = Field(
        description='The classified input type: "code", "text", or "problem_solving"'
    )


class ScoreOutput(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the score")
    score: float = Field(description="Score between 0.0 and 1.0")
    issues: list[str] = Field(
        description="List of specific issues, bugs, or suggestions found. Empty list if none."
    )
    verdict: str = Field(
        description="One-line summary verdict of the evaluation"
    )


class FeaturesOutput(BaseModel):
    features: str = Field(
        description="Key features extracted from the input relevant for evaluation"
    )


# ---------- Pipeline Steps ----------


async def step1_classify_input(input_data: str) -> str:
    """Step 1: Classify input as code, text, or problem_solving."""
    model = openai_plan_to_model_name["text-mini"]
    messages = compile_prompt(
        CLASSIFICATION_SYSTEM_PROMPT,
        CLASSIFICATION_USER_PROMPT,
        input_data=input_data,
    )

    result = await run_llm_with_openai(
        model=model,
        messages=messages,
        response_model=ClassificationOutput,
        max_output_tokens=512,
    )

    input_type = result.input_type.lower().strip()

    # Normalize to valid values
    valid_types = {"code", "text", "problem_solving"}
    if input_type not in valid_types:
        if "code" in input_type:
            input_type = "code"
        elif "problem" in input_type or "solving" in input_type:
            input_type = "problem_solving"
        else:
            input_type = "text"

    logger.info(f"MMEE Step 1 — Classified as: {input_type}")
    return input_type


async def step2_extract_features(input_data: str, input_type: str) -> str:
    """Step 2: Extract relevant features based on input type."""
    model = openai_plan_to_model_name["text-mini"]

    if input_type == "code":
        feature_prompt = (
            "Analyze this code and extract:\n"
            "- Programming language\n"
            "- Key logic/algorithm used\n"
            "- Potential bugs or syntax errors\n"
            "- Code structure (functions, classes, etc.)\n"
            "- Edge case handling\n"
            "- Time/space complexity estimate\n\n"
        
            "Scoring Rule:\n"
            "- If any syntax error is present, the code cannot execute.\n"
            "- In such cases, assign a score very close to zero but not exactly zero.\n"
            "- The score should reflect minimal credit for intent/structure if detectable.\n"
            "- If no syntax errors are present, evaluate normally based on logic, structure, and efficiency."
        )
    elif input_type == "problem_solving":
        feature_prompt = (
            "Analyze this problem-solving response and extract:\n"
            "- Problem being solved\n"
            "- Number of reasoning steps\n"
            "- Mathematical/logical operations used\n"
            "- Whether assumptions are stated\n"
            "- Whether the final answer is clearly stated\n"
            "- Logical flow quality"
        )
    else:
        feature_prompt = (
            "Analyze this text and extract:\n"
            "- Main topic/thesis\n"
            "- Structure (paragraphs, sections, etc.)\n"
            "- Key claims or arguments made\n"
            "- Depth of coverage\n"
            "- Factual claims that can be verified\n"
            "- Writing style and tone"
        )

    messages = [
        {"role": "system", "content": feature_prompt},
        {"role": "user", "content": input_data},
    ]

    result = await run_llm_with_openai(
        model=model,
        messages=messages,
        response_model=FeaturesOutput,
        max_output_tokens=1024,
    )

    logger.info(f"MMEE Step 2 — Features extracted")
    return result.features


async def step3_score(
    input_data: str, input_type: str, features: str
) -> tuple[float, float, float, dict]:
    """Step 3: Run three evaluators in parallel. Returns scores + feedback."""
    model = openai_plan_to_model_name["text"]

    # Build messages for each evaluator
    auto_messages = compile_prompt(
        AUTO_SCORE_SYSTEM_PROMPT,
        AUTO_SCORE_USER_PROMPT,
        input_type=input_type,
        input_data=input_data,
        features=features,
    )
    ai_messages = compile_prompt(
        AI_SCORE_SYSTEM_PROMPT,
        AI_SCORE_USER_PROMPT,
        input_type=input_type,
        input_data=input_data,
        features=features,
    )
    human_messages = compile_prompt(
        HUMAN_SCORE_SYSTEM_PROMPT,
        HUMAN_SCORE_USER_PROMPT,
        input_type=input_type,
        input_data=input_data,
        features=features,
    )

    # Run all three in parallel
    auto_task = run_llm_with_openai(
        model=model,
        messages=auto_messages,
        response_model=ScoreOutput,
        max_output_tokens=1024,
    )
    ai_task = run_llm_with_openai(
        model=model,
        messages=ai_messages,
        response_model=ScoreOutput,
        max_output_tokens=1024,
    )
    human_task = run_llm_with_openai(
        model=model,
        messages=human_messages,
        response_model=ScoreOutput,
        max_output_tokens=1024,
    )

    auto_result, ai_result, human_result = await asyncio.gather(
        auto_task, ai_task, human_task
    )

    # Clamp scores to [0, 1]
    a = max(0.0, min(1.0, auto_result.score))
    ai = max(0.0, min(1.0, ai_result.score))
    h = max(0.0, min(1.0, human_result.score))

    # Collect feedback from each evaluator
    feedback = {
        "auto": {
            "verdict": auto_result.verdict,
            "issues": auto_result.issues,
        },
        "ai": {
            "verdict": ai_result.verdict,
            "issues": ai_result.issues,
        },
        "human": {
            "verdict": human_result.verdict,
            "issues": human_result.issues,
        },
    }

    logger.info(f"MMEE Step 3 — Scores: auto={a:.2f}, ai={ai:.2f}, human={h:.2f}")
    logger.info(
        f"MMEE Step 3 — Issues: auto={len(auto_result.issues)}, "
        f"ai={len(ai_result.issues)}, human={len(human_result.issues)}"
    )
    return a, ai, h, feedback


async def step3_score_code(
    input_data: str, question_id: str, features: str, language: str = "python"
) -> tuple[float, float, float, dict, list[dict]]:
    """Step 3 for code mode: test execution for auto, LLM for AI + Human in parallel."""
    model = openai_plan_to_model_name["text"]
    test_cases = FIXED_QUESTIONS[question_id]["test_cases"]
    q_title = FIXED_QUESTIONS[question_id]["title"]

    ai_messages = compile_prompt(
        AI_SCORE_SYSTEM_PROMPT,
        AI_SCORE_USER_PROMPT,
        input_type="code",
        input_data=input_data,
        features=features,
    )
    human_messages = compile_prompt(
        HUMAN_SCORE_SYSTEM_PROMPT,
        HUMAN_SCORE_USER_PROMPT,
        input_type="code",
        input_data=input_data,
        features=features,
    )

    # Run test execution + AI + Human in parallel
    test_task = execute_test_cases(input_data, language, test_cases)
    ai_task = run_llm_with_openai(
        model=model, messages=ai_messages, response_model=ScoreOutput, max_output_tokens=1024
    )
    human_task = run_llm_with_openai(
        model=model, messages=human_messages, response_model=ScoreOutput, max_output_tokens=1024
    )

    test_results, ai_result, human_result = await asyncio.gather(test_task, ai_task, human_task)

    # Auto score from test execution
    passed = sum(1 for r in test_results if r.passed)
    total = len(test_results)
    auto_score = round(passed / total, 4) if total > 0 else 0.0
    auto_issues = [
        f"Test {r.test_case_id} {r.status}" + (f": {r.error}" if r.error else "")
        for r in test_results
        if not r.passed
    ]

    feedback = {
        "auto": {
            "verdict": f"{passed}/{total} test cases passed on '{q_title}'",
            "issues": auto_issues,
        },
        "ai": {"verdict": ai_result.verdict, "issues": ai_result.issues},
        "human": {"verdict": human_result.verdict, "issues": human_result.issues},
    }

    tc_results_data = [
        {
            "test_case_id": r.test_case_id,
            "input": r.input,
            "expected_output": r.expected_output,
            "actual_output": r.actual_output,
            "passed": r.passed,
            "status": r.status,
            "error": r.error,
        }
        for r in test_results
    ]

    logger.info(
        f"MMEE Step 3 (code) — auto={auto_score:.2f} ({passed}/{total} tests), "
        f"ai={ai_result.score:.2f}, human={human_result.score:.2f}"
    )
    return (
        auto_score,
        max(0.0, min(1.0, ai_result.score)),
        max(0.0, min(1.0, human_result.score)),
        feedback,
        tc_results_data,
    )


def step4_conflict(a: float, ai: float, h: float) -> float:
    """Step 4: Compute disagreement between evaluators."""
    conflict = max(a, ai, h) - min(a, ai, h)
    logger.info(f"MMEE Step 4 — Conflict: {conflict:.3f}")
    return round(conflict, 4)


def step5_weights(input_type: str, conflict: float) -> tuple[float, float, float]:
    """Step 5: Assign dynamic weights based on input type and conflict."""
    if input_type == "code":
        w_a, w_ai, w_h = 0.6, 0.3, 0.1
    elif input_type == "problem_solving":
        w_a, w_ai, w_h = 0.3, 0.4, 0.3
    else:  # text
        w_a, w_ai, w_h = 0.2, 0.4, 0.4

    # Adjust if conflict is high
    if conflict > 0.3:
        w_h += 0.2

    # Normalize
    total = w_a + w_ai + w_h
    w_a = round(w_a / total, 4)
    w_ai = round(w_ai / total, 4)
    w_h = round(w_h / total, 4)

    logger.info(
        f"MMEE Step 5 — Weights: auto={w_a}, ai={w_ai}, human={w_h} (conflict={conflict:.3f})"
    )
    return w_a, w_ai, w_h


def step6_trust_score(
    a: float, ai: float, h: float, w_a: float, w_ai: float, w_h: float
) -> float:
    """Step 6: Compute weighted trust score."""
    t = (w_a * a) + (w_ai * ai) + (w_h * h)
    logger.info(f"MMEE Step 6 — Trust score: {t:.4f}")
    return round(t, 4)


def step7_final_score(trust: float, conflict: float) -> tuple[float, float]:
    """Step 7: Apply conflict penalty and compute confidence."""
    s = trust - (0.2 * conflict)
    s = max(0.0, min(1.0, s))  # Clamp
    confidence = 1.0 - conflict
    confidence = max(0.0, min(1.0, confidence))
    logger.info(f"MMEE Step 7 — Final score: {s:.4f}, Confidence: {confidence:.4f}")
    return round(s, 4), round(confidence, 4)


def build_explanation(
    input_type: str,
    a: float,
    ai: float,
    h: float,
    conflict: float,
    w_a: float,
    w_ai: float,
    w_h: float,
    final_score: float,
) -> list[str]:
    """Build human-readable explanation bullets."""
    explanations = []

    # Input type context
    explanations.append(
        f"Input classified as '{input_type}' — weights adjusted accordingly "
        f"(auto={w_a:.2f}, ai={w_ai:.2f}, human={w_h:.2f})"
    )

    # Score analysis
    scores = {"Auto (correctness)": a, "AI (depth)": ai, "Human (readability)": h}
    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    explanations.append(
        f"Strongest dimension: {best} at {scores[best]:.2f}; "
        f"weakest: {worst} at {scores[worst]:.2f}"
    )

    # Conflict analysis
    if conflict > 0.3:
        explanations.append(
            f"High evaluator disagreement (conflict={conflict:.2f}) — "
            f"human weight boosted and penalty of {0.2 * conflict:.3f} applied to final score"
        )
    elif conflict > 0.15:
        explanations.append(
            f"Moderate evaluator agreement (conflict={conflict:.2f}) — "
            f"small penalty of {0.2 * conflict:.3f} applied"
        )
    else:
        explanations.append(
            f"Strong evaluator consensus (conflict={conflict:.2f}) — "
            f"minimal penalty, high confidence"
        )

    return explanations


# ---------- API Endpoints ----------


@router.get("/questions")
async def get_fixed_questions():
    """Return the list of fixed coding questions for the MMEE autograder."""
    return [
        {
            "id": q["id"],
            "title": q["title"],
            "description": q["description"],
            "input_format": q["input_format"],
            "output_format": q["output_format"],
            "templates": q["templates"],
            "test_cases": [
                {"id": tc["id"], "input": tc["input"], "expected_output": tc["expected_output"]}
                for tc in q["test_cases"]
            ],
        }
        for q in FIXED_QUESTIONS.values()
    ]


@router.post("")
async def evaluate_input(request: EvaluationRequest):
    """Main MMEE evaluation endpoint — runs the full 8-step pipeline."""
    input_data = request.input_data.strip()
    question_id = request.question_id
    language = request.language or "python"
    use_fixed_question = question_id and question_id in FIXED_QUESTIONS

    if not input_data:
        raise HTTPException(status_code=400, detail="input_data cannot be empty")

    async def stream_response() -> AsyncGenerator[str, None]:
        # Step 1: Classify (or use "code" if a fixed question is selected)
        yield json.dumps({"step": 1, "status": "classifying"}) + "\n"
        if use_fixed_question:
            input_type = "code"
        else:
            input_type = await step1_classify_input(input_data)
        yield json.dumps({"step": 1, "result": input_type}) + "\n"

        # Step 2: Extract features
        yield json.dumps({"step": 2, "status": "extracting_features"}) + "\n"
        features = await step2_extract_features(input_data, input_type)
        yield json.dumps({"step": 2, "result": "done"}) + "\n"

        # Step 3: Score
        yield json.dumps({"step": 3, "status": "scoring"}) + "\n"
        test_results_data: list[dict] | None = None

        if use_fixed_question:
            # Test execution for auto, LLM for AI + Human in parallel
            a, ai_score, h, feedback, test_results_data = await step3_score_code(
                input_data, question_id, features, language  # type: ignore[arg-type]
            )
        else:
            a, ai_score, h, feedback = await step3_score(input_data, input_type, features)

        step3_result: dict = {"auto": a, "ai": ai_score, "human": h}
        if test_results_data is not None:
            step3_result["test_results"] = test_results_data

        yield json.dumps({"step": 3, "result": step3_result}) + "\n"

        # Steps 4-7: Compute final result
        conflict = step4_conflict(a, ai_score, h)
        w_a, w_ai, w_h = step5_weights(input_type, conflict)
        trust = step6_trust_score(a, ai_score, h, w_a, w_ai, w_h)
        final_score, confidence = step7_final_score(trust, conflict)

        explanation = build_explanation(
            input_type, a, ai_score, h, conflict, w_a, w_ai, w_h, final_score
        )

        # Build feedback objects
        eval_feedback = EvaluationFeedback(
            auto=EvaluatorFeedback(**feedback["auto"]),
            ai=EvaluatorFeedback(**feedback["ai"]),
            human=EvaluatorFeedback(**feedback["human"]),
        )

        # Step 8: Build final response
        result = EvaluationResponse(
            input_type=input_type,
            scores=EvaluationScores(auto=a, ai=ai_score, human=h),
            weights=EvaluationWeights(auto=w_a, ai=w_ai, human=w_h),
            conflict=conflict,
            final_score=final_score,
            confidence=confidence,
            explanation=explanation,
            feedback=eval_feedback,
        )

        # Store in database
        try:
            await create_evaluation(
                input_type=input_type,
                input_data=input_data,
                auto_score=a,
                ai_score=ai_score,
                human_score=h,
                auto_weight=w_a,
                ai_weight=w_ai,
                human_weight=w_h,
                conflict=conflict,
                final_score=final_score,
                confidence=confidence,
                explanation=explanation,
                feedback=feedback,
            )
        except Exception as e:
            logger.error(f"Failed to store evaluation: {e}")

        yield json.dumps({"step": 8, "result": result.model_dump()}) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )


@router.get("/history")
async def evaluation_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Retrieve past evaluation results."""
    results = await get_evaluations(limit=limit, offset=offset)
    return results


@router.get("/{evaluation_id}")
async def get_single_evaluation(evaluation_id: int):
    """Retrieve a single evaluation by ID."""
    result = await get_evaluation_by_id(evaluation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result


# ---------- Code Evaluation Endpoints ----------

@router.post("/code", response_model=CodeEvaluationResponse)
async def evaluate_code(request: CodeEvaluationRequest):
    """
    Evaluate Python code against test cases.
    Optionally fetches from a saved draft if source_code is not provided.
    """
    source_code = request.source_code
    
    # 1. Fetch code from draft if not provided
    if not source_code:
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="Either source_code or user_id must be provided"
            )
        draft = await get_user_code_draft(request.user_id, request.question_id)
        if not draft or not draft.get("code"):
            raise HTTPException(status_code=404, detail="Saved draft not found")
            
        # Find the requested language code
        lang_code = next(
            (item for item in draft["code"] if item["language"] == request.language),
            None
        )
        if not lang_code:
            raise HTTPException(
                status_code=404,
                detail=f"No draft found for language: {request.language}"
            )
        source_code = lang_code["value"]

    # 2. Fetch test cases for the question
    test_cases = await get_test_cases_for_question(request.question_id)
    if not test_cases:
        raise HTTPException(
            status_code=404,
            detail=f"No test cases found for question {request.question_id}"
        )

    # 3. Execute code against test cases
    results = await execute_test_cases(source_code, request.language, test_cases)

    # 4. Compute strict score
    scoring_result = compute_score(results)

    # 5. Format total execution time
    total_time_ms = sum(r.execution_time_ms for r in results)
    formatted_time = f"{total_time_ms:.2f}ms"

    # 6. Save evaluation to history
    response_results = [
        CodeTestCaseResult(
            test_case_id=r.test_case_id,
            input=r.input,
            expected_output=r.expected_output,
            actual_output=r.actual_output,
            passed=r.passed,
            status=r.status,
            error=r.error,
            execution_time_ms=r.execution_time_ms,
        ) 
        for r in results
    ]

    eval_id = await create_code_evaluation(
        user_id=request.user_id,
        question_id=request.question_id,
        language=request.language,
        source_code=source_code,
        score=scoring_result.score,
        passed_testcases=scoring_result.passed_testcases,
        total_testcases=scoring_result.total_testcases,
        status=scoring_result.status,
        errors=scoring_result.errors,
        results=[r.model_dump() for r in response_results],
        execution_time_ms=total_time_ms,
    )

    from datetime import datetime, timezone
    
    return CodeEvaluationResponse(
        evaluation_id=eval_id,
        score=scoring_result.score,
        passed_testcases=scoring_result.passed_testcases,
        total_testcases=scoring_result.total_testcases,
        status=scoring_result.status,
        errors=scoring_result.errors,
        results=response_results,
        execution_time=formatted_time,
        submitted_at=datetime.now(timezone.utc).isoformat()
    )
