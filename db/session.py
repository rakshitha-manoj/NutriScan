"""MongoDB connection manager using PyMongo's native AsyncMongoClient.

Provides a singleton-style manager that integrates with FastAPI's lifespan.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymongo import AsyncMongoClient

if TYPE_CHECKING:
    from pymongo.asynchronous.database import AsyncDatabase

logger = logging.getLogger(__name__)

_DEFAULT_MONGO_URL = "mongodb://localhost:27017"
_DEFAULT_DB_NAME = "nutriscan"


class MongoDBManager:
    """Manages the async MongoDB connection lifecycle.

    Usage with FastAPI lifespan::

        db_manager = MongoDBManager()
        await db_manager.connect("mongodb://localhost:27017", "nutriscan")
        # ... app runs ...
        await db_manager.disconnect()
    """

    def __init__(self) -> None:
        self._client: AsyncMongoClient | None = None  # type: ignore[type-arg]
        self._db: AsyncDatabase | None = None  # type: ignore[type-arg]

    async def connect(
        self,
        url: str = _DEFAULT_MONGO_URL,
        db_name: str = _DEFAULT_DB_NAME,
    ) -> None:
        """Open the async MongoDB connection."""
        logger.info("Connecting to MongoDB at %s (db=%s)", url, db_name)
        self._client = AsyncMongoClient(url)
        self._db = self._client[db_name]
        logger.info("MongoDB connection established.")

    async def disconnect(self) -> None:
        """Close the async MongoDB connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed.")

    def get_database(self) -> AsyncDatabase:  # type: ignore[type-arg]
        """Return the current database handle.

        Raises:
            RuntimeError: If called before ``connect()``.
        """
        if self._db is None:
            raise RuntimeError("MongoDB is not connected. Call connect() first.")
        return self._db

    @property
    def is_connected(self) -> bool:
        """Check whether the client is initialised."""
        return self._client is not None


# Module-level singleton used across the application.
db_manager = MongoDBManager()
