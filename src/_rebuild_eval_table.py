import asyncio
from api.utils.db import get_new_db_connection
from api.db import create_evaluations_table

async def rebuild():
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute("DROP TABLE IF EXISTS evaluations")
        await conn.commit()
        print("Dropped old table")
        
        await create_evaluations_table(cursor)
        await conn.commit()
        print("Recreated with feedback column")

asyncio.run(rebuild())
