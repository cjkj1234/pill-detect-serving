"""ONNX Runtime 기반 알약 탐지기."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .classes import PillClass, load_class_mapping
from .config import Settings
from .postprocess import PostprocessConfig, postprocess
from .preprocess import preprocess


@dataclass
class Candidate:
    pill: PillClass
    prob: float


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # 원본 이미지 픽셀 좌표 x1, y1, x2, y2
    score: float
    pill: PillClass
    candidates: list[Candidate]


@dataclass
class PredictionResult:
    width: int
    height: int
    detections: list[Detection]
    timing_ms: dict[str, float] = field(default_factory=dict)


class Detector:
    """ONNX 모델 하나와 클래스 매핑을 들고, 이미지 → 탐지 결과를 만든다."""

    def __init__(
        self,
        model_path: str | Path,
        classes: list[PillClass],
        postprocess_config: PostprocessConfig | None = None,
        top_k_candidates: int = 3,
        intra_op_threads: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX 모델 파일이 없습니다: {self.model_path}")

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape  # [1, 3, S, S]
        if len(shape) != 4 or not all(isinstance(d, int) for d in shape[2:]) or shape[2] != shape[3]:
            raise ValueError(f"정사각형 고정 입력([1, 3, S, S]) ONNX만 지원합니다. 입력 shape: {shape}")
        self.input_size = int(shape[2])

        output_shape = self.session.get_outputs()[0].shape
        if len(output_shape) == 3 and isinstance(output_shape[1], int):
            model_classes = output_shape[1] - 4
            if model_classes != len(classes):
                raise ValueError(
                    f"모델 클래스 수({model_classes})와 클래스 매핑 수({len(classes)})가 다릅니다."
                )

        self.classes = classes
        self.postprocess_config = postprocess_config or PostprocessConfig()
        self.top_k_candidates = top_k_candidates

    @classmethod
    def from_settings(cls, settings: Settings) -> "Detector":
        return cls(
            model_path=settings.model_path,
            classes=load_class_mapping(settings.class_mapping_path),
            postprocess_config=PostprocessConfig(
                conf_threshold=settings.conf_threshold,
                iou_threshold=settings.iou_threshold,
                max_detections=settings.max_detections,
                class_agnostic=settings.class_agnostic_nms,
            ),
            top_k_candidates=settings.top_k_candidates,
            intra_op_threads=settings.intra_op_threads,
        )

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def predict(self, image_rgb: np.ndarray) -> PredictionResult:
        """RGB uint8 [H, W, 3] 이미지 1장을 탐지한다. 박스는 원본 이미지 좌표로 돌려준다."""
        height, width = image_rgb.shape[:2]

        t0 = time.perf_counter()
        inputs = preprocess(image_rgb, self.input_size)
        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: inputs})[0]
        t2 = time.perf_counter()
        raw_detections = postprocess(raw, self.postprocess_config)

        # 모델 입력(S×S) 좌표 → 원본 이미지 좌표. 정사각형으로 늘려 리사이즈했으므로 축마다 따로 되돌린다.
        scale = np.array(
            [width / self.input_size, height / self.input_size] * 2, dtype=np.float32
        )
        boxes = raw_detections.boxes * scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

        detections = []
        for box, score, label, probs in zip(
            boxes, raw_detections.scores, raw_detections.labels, raw_detections.class_probs
        ):
            top = np.argsort(-probs, kind="stable")[: self.top_k_candidates]
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    score=float(score),
                    pill=self.classes[int(label)],
                    candidates=[Candidate(pill=self.classes[int(i)], prob=float(probs[i])) for i in top],
                )
            )
        t3 = time.perf_counter()

        return PredictionResult(
            width=int(width),
            height=int(height),
            detections=detections,
            timing_ms={
                "preprocess": (t1 - t0) * 1000,
                "inference": (t2 - t1) * 1000,
                "postprocess": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
            },
        )
