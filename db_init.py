"""
Database initialization module for enhanced blockchain.
"""

import os
import asyncio
import logging
from postgres_storage import initialize_database

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize the PostgreSQL database schema."""
    try:
        logger.info("Initializing database schema...")
        await initialize_database()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def main():
    """Main entry point for database initialization."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run database initialization
    try:
        asyncio.run(init_database())
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
