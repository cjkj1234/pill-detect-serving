"""API 응답 스키마."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Candidate(BaseModel):
    class_id: int
    category_id: int
    name: str
    prob: float = Field(description="같은 박스 안에서의 클래스 확률 (후보끼리 합치면 최대 1)")


class Detection(BaseModel):
    box: Box = Field(description="원본 이미지 픽셀 좌표")
    score: float = Field(description="1위 클래스의 신뢰도 (0~1)")
    class_id: int
    category_id: int
    name: str
    candidates: list[Candidate] = Field(description="확률이 높은 순서의 후보 클래스")


class ImageInfo(BaseModel):
    width: int
    height: int


class Timing(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    image: ImageInfo
    detections: list[Detection]
    timing: Timing
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    detail: str | None = None


class ModelInfo(BaseModel):
    model_file: str
    input_size: int
    num_classes: int
    providers: list[str]
    conf_threshold: float
    iou_threshold: float
    max_detections: int
    class_agnostic_nms: bool
    disclaimer: str
