"""
Database Migration Script — Add User Memory Graph Tables

This script adds the user memory, preferences, and session tables to the database.

Usage:
    cd backend && python -m database.migrate_add_memory_tables
    # OR
    python -c "from database.migrate_add_memory_tables import create_tables; import asyncio; asyncio.run(create_tables())"

Requires:
    - POSTGRES_URL or DATABASE_URL environment variable
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sqlalchemy import text
from database.connection import engine
from database.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_tables():
    """Create all new tables defined in models.py"""
    try:
        logger.info("Starting database migration...")
        
        # This will create all tables defined in Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✓ Tables created successfully")
        
        # Verify tables exist
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname='public'
                ORDER BY tablename
            """))
            tables = result.fetchall()
            logger.info(f"✓ Current tables in database: {[t[0] for t in tables]}")
            
            # Check for new tables
            new_tables = ['user_memory', 'user_preference', 'user_session']
            for new_table in new_tables:
                has_table = any(new_table in t[0] for t in tables)
                status = "✓" if has_table else "✗"
                logger.info(f"  {status} {new_table}")
        
        logger.info("Migration complete!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(create_tables())
