"""환경 변수로 바꿀 수 있는 서빙 설정.

기본값은 PillDetect configs/*.yaml의 postprocess 섹션과 같다
(conf 0.25, IoU 0.7, 이미지당 최대 4개, class-agnostic NMS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("models/model.onnx")
    class_mapping_path: Path = Path("models/class_mapping.json")
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 4
    class_agnostic_nms: bool = True
    top_k_candidates: int = 3
    max_upload_mb: float = 10.0
    intra_op_threads: int = 0  # 0이면 ONNX Runtime 기본값

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        default = cls()
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", default.model_path)),
            class_mapping_path=Path(os.environ.get("CLASS_MAPPING_PATH", default.class_mapping_path)),
            conf_threshold=float(os.environ.get("CONF_THRESHOLD", default.conf_threshold)),
            iou_threshold=float(os.environ.get("IOU_THRESHOLD", default.iou_threshold)),
            max_detections=int(os.environ.get("MAX_DETECTIONS", default.max_detections)),
            class_agnostic_nms=_env_bool("CLASS_AGNOSTIC_NMS", default.class_agnostic_nms),
            top_k_candidates=int(os.environ.get("TOP_K_CANDIDATES", default.top_k_candidates)),
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", default.max_upload_mb)),
            intra_op_threads=int(os.environ.get("INTRA_OP_THREADS", default.intra_op_threads)),
        )
