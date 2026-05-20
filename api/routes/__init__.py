"""API routes package."""

from api.routes.fridge import router as fridge_router
from api.routes.health import router as health_router
from api.routes.plan import router as plan_router

__all__ = ["fridge_router", "health_router", "plan_router"]
