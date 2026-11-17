"""개발·CI용 더미 ONNX 모델을 만든다 (실제 가중치 없이 API·Docker를 끝까지 검증하는 용도).

    uv run python scripts/make_dummy_model.py --output models/model.onnx

클래스 수는 클래스 매핑 파일에서 읽는다. 입력과 상관없이 박스 3개(클래스 0, 2, 0)를 내는 모델이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pill_serving.classes import load_class_mapping
from pill_serving.dummy import build_dummy_onnx, example_raw_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--class-mapping", type=Path, default=Path("models/class_mapping.json"))
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    num_classes = len(load_class_mapping(args.class_mapping))
    path = build_dummy_onnx(args.output, example_raw_output(num_classes, args.input_size), args.input_size)
    print(f"더미 모델 저장: {path} (클래스 {num_classes}개, 입력 {args.input_size}px)")


if __name__ == "__main__":
    main()
