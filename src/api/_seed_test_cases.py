import asyncio
from api.utils.db import get_new_db_connection
from api.db.code_evaluation import create_test_case

async def seed():
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        
        question_id = 1
            
        test_cases = [
            {
                "input": "5\n1 2 3 4 5",
                "expected_output": "15",
                "is_hidden": False,
                "position": 1
            },
            {
                "input": "3\n10 20 30",
                "expected_output": "60",
                "is_hidden": False,
                "position": 2
            },
            {
                "input": "0\n",
                "expected_output": "0",
                "is_hidden": True,
                "position": 3
            }
        ]
        
        for tc in test_cases:
            tc_id = await create_test_case(
                question_id=question_id,
                input_data=tc["input"],
                expected_output=tc["expected_output"],
                is_hidden=tc["is_hidden"],
                position=tc["position"]
            )
            print(f"Created test case {tc_id}")

        print("Done seeding test cases.")

if __name__ == "__main__":
    asyncio.run(seed())
