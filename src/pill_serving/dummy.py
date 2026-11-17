"""개발·테스트·CI용 더미 ONNX 모델.

실제 YOLO 가중치 없이도 API와 Docker 이미지를 끝까지 검증할 수 있게,
입력과 상관없이 정해진 raw output [1, 4 + nc, num_anchors]를 내는 ONNX 그래프를 만든다.
(`onnx` 패키지가 필요하다: dev 의존성 그룹)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def example_raw_output(num_classes: int, input_size: int = 640) -> np.ndarray:
    """[4 + nc, 5] raw output. 열 하나가 앵커 하나이고, 좌표는 input_size 기준 픽셀이다.

    앵커 0: 박스 A, 클래스 0 (0.90)
    앵커 1: 박스 A와 거의 같은 위치, 클래스 0 (0.80) → NMS에서 제거
    앵커 2: 박스 B, 클래스 0 (0.60), 2위 클래스 1 (0.35) — A와 같은 클래스
    앵커 3: 박스 C, 클래스 2 (0.70)
    앵커 4: 모든 점수가 낮아 conf threshold에서 제거
    """
    if num_classes < 3:
        raise ValueError("num_classes는 3 이상이어야 합니다.")
    s = input_size / 640
    boxes = np.array(
        [
            [50, 50, 20, 20],
            [51, 50, 20, 20],
            [200, 200, 20, 20],
            [400, 100, 30, 30],
            [300, 300, 10, 10],
        ],
        dtype=np.float32,
    ) * s
    scores = np.full((5, num_classes), 0.001, dtype=np.float32)
    scores[0, :3] = [0.90, 0.05, 0.05]
    scores[1, :3] = [0.80, 0.10, 0.05]
    scores[2, :3] = [0.60, 0.35, 0.05]
    scores[3, :3] = [0.02, 0.03, 0.70]
    scores[4, :3] = [0.01, 0.01, 0.01]
    return np.concatenate([boxes, scores], axis=1).T.copy()


def build_dummy_onnx(path: str | Path, raw_output: np.ndarray, input_size: int = 640) -> Path:
    """입력 [1, 3, S, S]를 받아 항상 raw_output을 내는 ONNX 파일을 만든다."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    raw = np.asarray(raw_output, dtype=np.float32)[None]  # [1, 4 + nc, A]
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, input_size, input_size])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))

    # 입력을 그래프에 연결하기 위해 mean(images) * 0 을 더한다 (결과는 상수 그대로).
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["images_mean"], keepdims=0),
        helper.make_node("Mul", ["images_mean", "zero"], ["images_zero"]),
        helper.make_node("Add", ["raw_const", "images_zero"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dummy_pill_detector",
        [images],
        [output],
        initializer=[
            numpy_helper.from_array(raw, name="raw_const"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="pill-serving-dummy")
    model.ir_version = 9  # 오래된 ONNX Runtime에서도 읽히도록 고정
    onnx.checker.check_model(model)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
