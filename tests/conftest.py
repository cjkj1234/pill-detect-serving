from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pill_serving.dummy import build_dummy_onnx, example_raw_output

NUM_CLASSES = 3
INPUT_SIZE = 640
CLASS_NAMES = ["알약가정 10mg", "알약나캡슐 5mg", "알약다정 2mg"]


@pytest.fixture(scope="session")
def raw_output() -> np.ndarray:
    return example_raw_output(NUM_CLASSES, INPUT_SIZE)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory, raw_output: np.ndarray) -> Path:
    """더미 ONNX(클래스 3개)와 클래스 매핑 파일이 든 폴더."""
    root = tmp_path_factory.mktemp("models")
    build_dummy_onnx(root / "model.onnx", raw_output, INPUT_SIZE)
    mapping = {
        "num_classes": NUM_CLASSES,
        "classes": [
            {"class_id": i, "category_id": 1000 + i, "name": name} for i, name in enumerate(CLASS_NAMES)
        ],
    }
    (root / "class_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return root


def make_png(width: int = 976, height: int = 1280) -> bytes:
    """PillDetect 원본과 같은 976×1280 크기의 회색 PNG."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return make_png()
