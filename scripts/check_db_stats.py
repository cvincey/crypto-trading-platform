"""Check database table and row counts."""

import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import text
from crypto.data.database import get_async_session


async def get_table_info():
    """Get table names and row counts from the database."""
    async with get_async_session() as session:
        # Get all table names
        result = await session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]
        
        print(f'\nTotal tables: {len(tables)}\n')
        print(f'{"Table Name":<25} {"Row Count":>15}')
        print('=' * 45)
        
        total_rows = 0
        for table in tables:
            count_result = await session.execute(text(f'SELECT COUNT(*) FROM {table}'))
            count = count_result.scalar()
            total_rows += count
            print(f'{table:<25} {count:>15,}')
        
        print('=' * 45)
        print(f'{"TOTAL":<25} {total_rows:>15,}')
        print()


if __name__ == '__main__':
    asyncio.run(get_table_info())

