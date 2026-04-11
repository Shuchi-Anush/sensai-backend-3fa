from fastapi import APIRouter, HTTPException
from typing import List

from api.models import CreateTestCaseRequest, TestCaseResponse
from api.db.code_evaluation import (
    create_test_case,
    get_test_cases_for_question,
    delete_test_case,
)

router = APIRouter()


@router.post("/", response_model=TestCaseResponse)
async def add_test_case(request: CreateTestCaseRequest):
    tc_id = await create_test_case(
        question_id=request.question_id,
        input_data=request.input,
        expected_output=request.expected_output,
        is_hidden=request.is_hidden,
        position=request.position,
    )
    return TestCaseResponse(
        id=tc_id,
        question_id=request.question_id,
        input=request.input,
        expected_output=request.expected_output,
        is_hidden=request.is_hidden,
        position=request.position,
    )


@router.get("/question/{question_id}", response_model=List[TestCaseResponse])
async def get_test_cases(question_id: int):
    tcs = await get_test_cases_for_question(question_id)
    return [TestCaseResponse(**tc) for tc in tcs]


@router.delete("/{test_case_id}")
async def remove_test_case(test_case_id: int):
    success = await delete_test_case(test_case_id)
    if not success:
        raise HTTPException(status_code=404, detail="Test case not found")
    return {"success": True}
