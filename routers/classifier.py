import base64
import binascii
import json
import os
import sys
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional

sys.path.append("..")

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import models
from database import SessionLocal
from detect_and_classify import (
    DEVICE,
    CONVNEXT_MODEL,
    YOLO_MODEL,
    CLASS_NAMES,
    detect_and_classify,
    load_classifier,
    load_detector,
)
from pathfinder import build_local_coach, generate_gemini_coach, normalize_holds
from PIL import Image


class ImagePayload(BaseModel):
    filename: str = Field(..., description="Original file name provided by the client")
    content_type: str = Field(..., description="MIME type of the image")
    data: str = Field(..., description="Base64-encoded image content")


class ImageResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int
    message: str
    classifications: List[Dict[str, float]]
    holds: Optional[List[Dict[str, object]]] = None
    jsonout: Optional[Dict[str, Any]] = None


class HoldPayload(BaseModel):
    id: Optional[int] = None
    bbox: Optional[List[float]] = None
    box: Optional[List[float]] = None
    type: Optional[str] = None
    class_name: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


class PathfinderPayload(BaseModel):
    image_id: int = Field(..., description="ID of the stored image to analyze")
    holds: Optional[List[HoldPayload]] = Field(None, description="Detection holds with bounding boxes")
    model: Optional[str] = Field(None, description="Gemini model to use when available")
    local_only: bool = Field(False, description="Skip Gemini and use heuristic pathfinder")


class PathfinderResponse(BaseModel):
    image_id: int
    coach: Dict[str, Any]
    
router = APIRouter(
    prefix="/classifier",
    tags=["classifier"],
    responses={404: {"description": "Not found"}},
)

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

def detect_results(detector, classifier, image_bytes: bytes, device: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        results = detect_and_classify(
            detector,
            classifier,
            tmp_path,
            device,
            save_output=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return results


def build_classifications(results: list) -> List[Dict[str, float]]:
    return [
        {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, det["probs"].tolist())}
        for det in results
    ]


def build_holds(results: list) -> List[Dict[str, object]]:
    holds: List[Dict[str, object]] = []
    for i, det in enumerate(results):
        box = det.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        holds.append(
            {
                "id": i,
                "bbox": [x1, y1, x2, y2],
                "type": det.get("class_name"),
                "confidence": float(det.get("confidence", 0.0)),
            }
        )
    return holds


@router.post("/upload", response_model=ImageResponse)
async def upload_image(payload: ImagePayload, db_session=Depends(get_db)):

    # upload and decode image step
    try:
        binary_content = base64.b64decode(payload.data, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc

    image = models.Image(
        filename=payload.filename,
        content_type=payload.content_type,
        data=binary_content,
    )

    # Persist image if desired
    db_session.add(image)
    db_session.commit()

    global DETECTOR_INSTANCE, CLASSIFIER_INSTANCE
    if "DETECTOR_INSTANCE" not in globals():
        DETECTOR_INSTANCE = load_detector(YOLO_MODEL)
    if "CLASSIFIER_INSTANCE" not in globals():
        CLASSIFIER_INSTANCE = load_classifier(CONVNEXT_MODEL, DEVICE)

    results = detect_results(
        DETECTOR_INSTANCE,
        CLASSIFIER_INSTANCE,
        binary_content,
        device=DEVICE,
    )

    classifications = build_classifications(results)
    holds = build_holds(results)


    # Return classification results, last step
    return ImageResponse(
        id=image.id,
        filename=image.filename,
        content_type=image.content_type,
        size=len(binary_content),
        message="Image uploaded successfully",
        classifications=classifications,
        holds=holds,
    )

@router.post("/pathfinder")
async def pathfinder(payload: PathfinderPayload, db_session=Depends(get_db)):
    image = db_session.get(models.Image, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    hold_items = [h.dict(exclude_unset=True) for h in payload.holds] if payload.holds else None

    if not hold_items:
        global DETECTOR_INSTANCE, CLASSIFIER_INSTANCE
        if "DETECTOR_INSTANCE" not in globals():
            DETECTOR_INSTANCE = load_detector(YOLO_MODEL)
        if "CLASSIFIER_INSTANCE" not in globals():
            CLASSIFIER_INSTANCE = load_classifier(CONVNEXT_MODEL, DEVICE)

        detection_results = detect_results(
            DETECTOR_INSTANCE,
            CLASSIFIER_INSTANCE,
            image.data,
            device=DEVICE,
        )
        hold_items = build_holds(detection_results)

        if not hold_items:
            if image.path_found is None:
                raise HTTPException(status_code=404, detail="No pathfinder data stored for this image")
            return PathfinderResponse(image_id=image.id, coach=image.path_found)

    try:
        img = Image.open(BytesIO(image.data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load stored image") from exc
    
    img_w, img_h = img.size

    try:
        normalized = normalize_holds({"holds": hold_items}, img_w, img_h)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    coach: Optional[Dict[str, Any]] = None
    if not payload.local_only:
        try:
            coach = generate_gemini_coach(img, normalized, payload.model or "models/gemini-2.5-flash")
        except Exception:
            coach = None

    if coach is None:
        coach = build_local_coach(normalized)

    image.path_found = coach
    db_session.add(image)
    db_session.commit()

    return PathfinderResponse(image_id=image.id, coach=coach)