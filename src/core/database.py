# ==============================================================================
# Database Connection Manager
# ==============================================================================
#
# Manages PostgreSQL/PGVector connection lifecycle with proper cleanup.
#
# This module provides:
#   1. Singleton pattern for API serving (reuse connections)
#   2. Context manager for batch jobs (automatic cleanup)
#   3. Proper async disposal of connection pools
#
# Usage (API serving - singleton):
#   db = DatabaseManager.get_instance(connection_string)
#   chain = RAGChain(pg_engine=db.engine, ...)
#
# Usage (Batch jobs - temporary):
#   with DatabaseManager.create_temporary(connection_string) as db:
#       chain = RAGChain(pg_engine=db.engine, ...)
#       # Connections automatically closed after context
#
# =============================================================================="""

from contextlib import contextmanager
from typing import Optional

from langchain_postgres import PGEngine

from src.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections lifecycle."""

    _instance: Optional["DatabaseManager"] = None

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        try:
            logger.info("Initializing DatabaseManager...")
            self.engine = PGEngine.from_connection_string(
                url=connection_string,
                connect_args={"connect_timeout": 10},
            )
            logger.info("✓ DatabaseManager initialized")
        except Exception as e:
            logger.error(f"Error initializing logger: {e}")

    @classmethod
    def get_instance(cls, connection_string: str) -> "DatabaseManager":
        """Singleton - reuse connections in serving."""
        if cls._instance is None:
            cls._instance = cls(connection_string)
        return cls._instance

    @classmethod
    @contextmanager
    def create_temporary(cls, connection_string: str):
        """New instance - for batch jobs that need cleanup."""
        manager = cls(connection_string)
        try:
            yield manager
        finally:
            manager.close()

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None

    def close(self):
        """Clean up database connections."""
        if self.engine:
            # PGEngine uses async engine internally, dispose the sync way
            import asyncio

            async def _dispose():
                await self.engine._pool.dispose()

            try:
                # Try to get existing loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule it if loop is running
                    asyncio.ensure_future(_dispose())
                else:
                    loop.run_until_complete(_dispose())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(_dispose())

            logger.info("✓ DatabaseManager connections closed")
