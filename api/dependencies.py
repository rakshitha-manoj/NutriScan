"""FastAPI dependency providers — singletons for DB, models, and config."""

from __future__ import annotations

import logging
from typing import Any

from api.settings import settings
from db.session import db_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons (created on first access, not at import time)
# ---------------------------------------------------------------------------

_freshness_instance: Any = None
_freshness_loaded: bool = False

_estimator_instance: Any = None
_estimator_loaded: bool = False


async def get_db() -> Any:
    """Yield the async MongoDB database handle."""
    return db_manager.get_database()


def get_freshness_inference() -> Any:
    """Return the module-level FreshnessInference singleton.

    Returns ``None`` if the checkpoint file does not exist (routes
    must handle this gracefully, e.g. by returning 503).
    """
    global _freshness_instance, _freshness_loaded  # noqa: PLW0603
    if not _freshness_loaded:
        _freshness_loaded = True
        path = settings.freshness_checkpoint_path
        if path.exists():
            from models.freshness.inference import FreshnessInference

            _freshness_instance = FreshnessInference(checkpoint_path=path)
            logger.info("FreshnessInference loaded from %s", path)
        else:
            logger.warning("Freshness checkpoint not found at %s", path)
    return _freshness_instance


def get_portion_estimator() -> Any:
    """Return the module-level PortionEstimator singleton.

    ``PortionEstimator`` initialises YOLO lazily so this is safe at
    import time.
    """
    global _estimator_instance, _estimator_loaded  # noqa: PLW0603
    if not _estimator_loaded:
        _estimator_loaded = True
        from models.portion.estimator import PortionEstimator

        _estimator_instance = PortionEstimator()
        logger.info("PortionEstimator initialised (YOLO loaded lazily).")
    return _estimator_instance
