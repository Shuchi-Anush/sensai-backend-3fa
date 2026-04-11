import json
from api.utils.db import get_new_db_connection
from api.config import evaluations_table_name
from api.utils.logging import logger


async def create_evaluation(
    input_type: str,
    input_data: str,
    auto_score: float,
    ai_score: float,
    human_score: float,
    auto_weight: float,
    ai_weight: float,
    human_weight: float,
    conflict: float,
    final_score: float,
    confidence: float,
    explanation: list[str],
    feedback: dict | None = None,
) -> int:
    """Store an evaluation result and return its ID."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""INSERT INTO {evaluations_table_name}
                (input_type, input_data, auto_score, ai_score, human_score,
                 auto_weight, ai_weight, human_weight, conflict,
                 final_score, confidence, explanation, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input_type,
                input_data,
                auto_score,
                ai_score,
                human_score,
                auto_weight,
                ai_weight,
                human_weight,
                conflict,
                final_score,
                confidence,
                json.dumps(explanation),
                json.dumps(feedback) if feedback else None,
            ),
        )
        await conn.commit()
        evaluation_id = cursor.lastrowid
        logger.info(f"Stored evaluation {evaluation_id} (type={input_type})")
        return evaluation_id


async def get_evaluations(limit: int = 20, offset: int = 0) -> list[dict]:
    """Retrieve evaluation history, most recent first."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""SELECT id, input_type, input_data, auto_score, ai_score, human_score,
                       auto_weight, ai_weight, human_weight, conflict,
                       final_score, confidence, explanation, feedback, created_at
                FROM {evaluations_table_name}
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in rows:
            item = dict(zip(columns, row))
            item["explanation"] = json.loads(item["explanation"])
            if item.get("feedback"):
                item["feedback"] = json.loads(item["feedback"])
            results.append(item)
        return results


async def get_evaluation_by_id(evaluation_id: int) -> dict | None:
    """Retrieve a single evaluation by ID."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""SELECT id, input_type, input_data, auto_score, ai_score, human_score,
                       auto_weight, ai_weight, human_weight, conflict,
                       final_score, confidence, explanation, feedback, created_at
                FROM {evaluations_table_name}
                WHERE id = ? AND deleted_at IS NULL""",
            (evaluation_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        item = dict(zip(columns, row))
        item["explanation"] = json.loads(item["explanation"])
        if item.get("feedback"):
            item["feedback"] = json.loads(item["feedback"])
        return item
