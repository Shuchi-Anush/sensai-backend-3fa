import asyncio
from api.utils.db import get_new_db_connection
from api.db import create_code_test_cases_table, create_code_evaluations_table

async def run():
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        print("Creating table: code_test_cases")
        await create_code_test_cases_table(cursor)
        print("Creating table: code_evaluations")
        await create_code_evaluations_table(cursor)
        await conn.commit()
    print("Tables created successfully.")

if __name__ == "__main__":
    asyncio.run(run())
