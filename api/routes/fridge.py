"""Fridge image analysis route."""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from PIL import Image

from api.dependencies import get_db, get_freshness_inference, get_portion_estimator
from api.schemas import DetectedItemResponse, FridgeAnalyseResponse
from db.models import BoundingBox, DetectedItem, FridgeState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fridge", tags=["fridge"])

_UPLOAD_DIR = Path("data/raw/uploads")
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

DbDep = Annotated[Any, Depends(get_db)]
EstimatorDep = Annotated[Any, Depends(get_portion_estimator)]
FreshnessDep = Annotated[Any, Depends(get_freshness_inference)]


@router.post("/analyse", response_model=FridgeAnalyseResponse)
async def analyse_fridge(
    image: UploadFile,
    user_id: Annotated[str, Form()],
    db: DbDep,
    estimator: EstimatorDep,
    freshness: FreshnessDep,
) -> FridgeAnalyseResponse:
    """Analyse a fridge image: detect items, estimate portions & freshness."""
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {image.content_type}. Use JPEG or PNG.",
        )

    if estimator is None:
        raise HTTPException(status_code=503, detail="Portion estimator unavailable.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg" if image.content_type == "image/jpeg" else ".png"
    tmp_path = _UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        content = await image.read()
        logger.info(
            "Received image upload: user_id=%s, size=%d bytes, path=%s",
            user_id,
            len(content),
            tmp_path,
        )
        tmp_path.write_bytes(content)

        portions = estimator.estimate(str(tmp_path))

        items: list[DetectedItemResponse] = []
        detected_items: list[DetectedItem] = []
        ratio = 0.0

        if portions:
            ratio = portions[0].pixel_to_cm2_ratio
            img = Image.open(tmp_path).convert("RGB")

            for portion in portions:
                f_score = 0.5
                f_unc = 0.5
                f_label = "unknown"

                if freshness is not None:
                    f_score, f_unc, f_label = _run_freshness(freshness, img, portion.bounding_box)

                bb = portion.bounding_box
                items.append(
                    DetectedItemResponse(
                        label=portion.label,
                        bounding_box={
                            "x": bb.x_min,
                            "y": bb.y_min,
                            "width": bb.x_max - bb.x_min,
                            "height": bb.y_max - bb.y_min,
                        },
                        detection_confidence=portion.detection_confidence,
                        estimated_grams=portion.estimated_grams,
                        uncertainty_grams=portion.uncertainty_grams,
                        freshness_score=f_score,
                        freshness_uncertainty=f_unc,
                        freshness_label=f_label,
                    )
                )
                detected_items.append(
                    DetectedItem(
                        name=portion.label,
                        bounding_box=bb,
                        confidence=portion.detection_confidence,
                        freshness_score=f_score,
                        estimated_grams=portion.estimated_grams,
                    )
                )

        now = datetime.now(tz=UTC)
        fridge_doc = FridgeState(
            user_id=user_id,
            image_path=str(tmp_path),
            detected_items=detected_items,
            captured_at=now,
        )
        await db.fridge_states.insert_one(fridge_doc.model_dump())

        return FridgeAnalyseResponse(
            user_id=user_id,
            captured_at=now,
            items=items,
            pixel_to_cm2_ratio=ratio,
            message=f"Detected {len(items)} item(s).",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing fridge image for user %s", user_id)
        raise HTTPException(status_code=500, detail="Internal server error.") from None
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_freshness(freshness: Any, img: Image.Image, bb: BoundingBox) -> tuple[float, float, str]:
    """Crop bounding box and run freshness inference."""
    crop = img.crop((int(bb.x_min), int(bb.y_min), int(bb.x_max), int(bb.y_max)))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        crop.save(tmp, format="JPEG")
        crop_path = Path(tmp.name)
    try:
        pred = freshness.predict(crop_path)
        return pred.freshness_score, pred.uncertainty, pred.label
    finally:
        crop_path.unlink(missing_ok=True)
