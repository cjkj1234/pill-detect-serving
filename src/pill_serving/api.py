"""FastAPI 서빙 앱.

    uvicorn pill_serving.api:app --host 0.0.0.0 --port 8000

모델 파일이 없어도 서버는 뜬다. /health가 model_loaded=false와 이유를 알려주고,
/predict는 503을 돌려준다. (docker compose에서 모델을 아직 넣지 않았을 때 원인을 바로 알 수 있게)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from . import DISCLAIMER, __version__
from .config import Settings
from .detector import Detector, PredictionResult
from .preprocess import InvalidImageError, load_image
from .schemas import (
    Box,
    Candidate,
    Detection,
    HealthResponse,
    ImageInfo,
    ModelInfo,
    PredictResponse,
    Timing,
)

logger = logging.getLogger("pill_serving")


def _to_response(result: PredictionResult) -> PredictResponse:
    return PredictResponse(
        image=ImageInfo(width=result.width, height=result.height),
        detections=[
            Detection(
                box=Box(x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3]),
                score=d.score,
                class_id=d.pill.class_id,
                category_id=d.pill.category_id,
                name=d.pill.name,
                candidates=[
                    Candidate(
                        class_id=c.pill.class_id,
                        category_id=c.pill.category_id,
                        name=c.pill.name,
                        prob=c.prob,
                    )
                    for c in d.candidates
                ],
            )
            for d in result.detections
        ],
        timing=Timing(
            preprocess_ms=result.timing_ms["preprocess"],
            inference_ms=result.timing_ms["inference"],
            postprocess_ms=result.timing_ms["postprocess"],
            total_ms=result.timing_ms["total"],
        ),
        disclaimer=DISCLAIMER,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = None
        app.state.load_error = None
        try:
            app.state.detector = Detector.from_settings(settings)
        except Exception as exc:  # 모델 파일 없음·형식 오류·클래스 수 불일치 등 — 서버는 띄우고 원인을 /health로 알린다
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("모델 로드 실패: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="PillDetect Serving",
        version=__version__,
        description="YOLO11 알약 탐지 모델(ONNX)을 서빙하는 API. " + DISCLAIMER,
        lifespan=lifespan,
    )

    def get_detector(request: Request) -> Detector:
        detector = request.app.state.detector
        if detector is None:
            raise HTTPException(status_code=503, detail=f"모델이 로드되지 않았습니다: {request.app.state.load_error}")
        return detector

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.detector is not None
        return HealthResponse(status="ok", model_loaded=loaded, detail=None if loaded else request.app.state.load_error)

    @app.get("/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        detector = get_detector(request)
        return ModelInfo(
            model_file=detector.model_path.name,
            input_size=detector.input_size,
            num_classes=len(detector.classes),
            providers=detector.providers,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
            class_agnostic_nms=settings.class_agnostic_nms,
            disclaimer=DISCLAIMER,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, file: UploadFile = File(..., description="알약 사진 (PNG/JPEG)")) -> PredictResponse:
        detector = get_detector(request)

        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다 (최대 {settings.max_upload_mb:g}MB).")
        if not data:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        try:
            image = load_image(data)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # ONNX Runtime 추론은 CPU 작업이라 이벤트 루프를 막지 않도록 스레드풀에서 실행한다.
        result = await run_in_threadpool(detector.predict, image)
        return _to_response(result)

    return app


app = create_app()
