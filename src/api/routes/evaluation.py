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


@router.post("")
async def evaluate_input(request: EvaluationRequest):
    """Main MMEE evaluation endpoint — runs the full 8-step pipeline."""
    input_data = request.input_data.strip()
    if not input_data:
        raise HTTPException(status_code=400, detail="input_data cannot be empty")

    async def stream_response() -> AsyncGenerator[str, None]:
        # Step 1: Classify
        yield json.dumps({"step": 1, "status": "classifying"}) + "\n"
        input_type = await step1_classify_input(input_data)
        yield json.dumps({"step": 1, "result": input_type}) + "\n"

        # Step 2: Extract features
        yield json.dumps({"step": 2, "status": "extracting_features"}) + "\n"
        features = await step2_extract_features(input_data, input_type)
        yield json.dumps({"step": 2, "result": "done"}) + "\n"

        # Step 3: Score (three parallel evaluators) — now returns feedback
        yield json.dumps({"step": 3, "status": "scoring"}) + "\n"
        a, ai_score, h, feedback = await step3_score(
            input_data, input_type, features
        )
        yield json.dumps(
            {"step": 3, "result": {"auto": a, "ai": ai_score, "human": h}}
        ) + "\n"

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
