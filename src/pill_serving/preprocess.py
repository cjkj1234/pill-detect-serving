"""이미지 전처리 — PillDetect 검증·추론 파이프라인과 같게 맞춘다.

PillDetect val_transform: PIL.Image.open(...).convert("RGB")
                          → albumentations.Resize(S, S)  (기본 보간 cv2.INTER_LINEAR)
                          → torchvision to_tensor        ([C, H, W], 0~1 float32)

리사이즈를 PIL로 바꾸면(축소 시 안티앨리어싱이 들어감) 학습 때와 픽셀 값이 달라져
정확도가 떨어질 수 있어서, 학습과 같은 cv2.INTER_LINEAR를 쓴다.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """업로드한 파일을 이미지로 읽을 수 없을 때."""


def load_image(data: bytes) -> np.ndarray:
    """이미지 바이트 → RGB uint8 배열 [H, W, 3].

    휴대폰 사진은 EXIF 회전 정보가 있어서 화면에 보이는 방향으로 먼저 돌린다.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise InvalidImageError("이미지 파일이 아니거나 지원하지 않는 형식입니다.") from exc
    except Exception as exc:  # 손상된 파일, 너무 큰 이미지(DecompressionBombError) 등 — 모두 잘못된 입력으로 처리
        raise InvalidImageError(f"이미지를 읽을 수 없습니다: {type(exc).__name__}") from exc


def preprocess(image_rgb: np.ndarray, input_size: int) -> np.ndarray:
    """RGB uint8 [H, W, 3] → 모델 입력 float32 [1, 3, S, S] (0~1)."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise InvalidImageError("RGB 3채널 이미지여야 합니다.")
    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
