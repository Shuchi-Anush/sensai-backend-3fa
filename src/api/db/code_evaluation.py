"""
Database operations for code test cases and code evaluations.
"""

import json
from api.utils.db import get_new_db_connection
from api.config import code_test_cases_table_name, code_evaluations_table_name
from api.utils.logging import logger


# ---------- Test Cases ----------


async def create_test_case(
    question_id: int,
    input_data: str,
    expected_output: str,
    is_hidden: bool = False,
    position: int = 0,
) -> int:
    """Create a test case for a question. Returns the test case ID."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""INSERT INTO {code_test_cases_table_name}
                (question_id, input, expected_output, is_hidden, position)
                VALUES (?, ?, ?, ?, ?)""",
            (question_id, input_data, expected_output, is_hidden, position),
        )
        await conn.commit()
        tc_id = cursor.lastrowid
        logger.info(f"Created test case {tc_id} for question {question_id}")
        return tc_id


async def get_test_cases_for_question(question_id: int) -> list[dict]:
    """Get all active test cases for a question, ordered by position."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""SELECT id, question_id, input, expected_output, is_hidden, position
                FROM {code_test_cases_table_name}
                WHERE question_id = ? AND deleted_at IS NULL
                ORDER BY position ASC""",
            (question_id,),
        )
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


async def delete_test_case(test_case_id: int) -> bool:
    """Soft-delete a test case."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""UPDATE {code_test_cases_table_name}
                SET deleted_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL""",
            (test_case_id,),
        )
        await conn.commit()
        return cursor.rowcount > 0


# ---------- Code Evaluations ----------


async def create_code_evaluation(
    user_id: int | None,
    question_id: int,
    language: str,
    source_code: str,
    score: int,
    passed_testcases: int,
    total_testcases: int,
    status: str,
    errors: list[str],
    results: list[dict],
    execution_time_ms: float,
) -> int:
    """Store a code evaluation result. Returns the evaluation ID."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""INSERT INTO {code_evaluations_table_name}
                (user_id, question_id, language, source_code, score,
                 passed_testcases, total_testcases, status, errors,
                 results, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                question_id,
                language,
                source_code,
                score,
                passed_testcases,
                total_testcases,
                status,
                json.dumps(errors),
                json.dumps(results),
                execution_time_ms,
            ),
        )
        await conn.commit()
        eval_id = cursor.lastrowid
        logger.info(
            f"Stored code evaluation {eval_id} — score={score}/10  "
            f"passed={passed_testcases}/{total_testcases}"
        )
        return eval_id


async def get_code_evaluations(
    question_id: int | None = None,
    user_id: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """Retrieve code evaluation history with optional filters."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()

        where_clauses = ["deleted_at IS NULL"]
        params: list = []

        if question_id is not None:
            where_clauses.append("question_id = ?")
            params.append(question_id)
        if user_id is not None:
            where_clauses.append("user_id = ?")
            params.append(user_id)

        where = " AND ".join(where_clauses)
        params.extend([limit, offset])

        await cursor.execute(
            f"""SELECT id, user_id, question_id, language, source_code,
                       score, passed_testcases, total_testcases, status,
                       errors, results, execution_time_ms, created_at
                FROM {code_evaluations_table_name}
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            tuple(params),
        )
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results_list = []
        for row in rows:
            item = dict(zip(columns, row))
            item["errors"] = json.loads(item["errors"]) if item["errors"] else []
            item["results"] = json.loads(item["results"]) if item["results"] else []
            results_list.append(item)
        return results_list


async def get_code_evaluation_by_id(evaluation_id: int) -> dict | None:
    """Retrieve a single code evaluation by ID."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""SELECT id, user_id, question_id, language, source_code,
                       score, passed_testcases, total_testcases, status,
                       errors, results, execution_time_ms, created_at
                FROM {code_evaluations_table_name}
                WHERE id = ? AND deleted_at IS NULL""",
            (evaluation_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        item = dict(zip(columns, row))
        item["errors"] = json.loads(item["errors"]) if item["errors"] else []
        item["results"] = json.loads(item["results"]) if item["results"] else []
        return item
