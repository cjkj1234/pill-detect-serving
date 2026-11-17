"""FP32 ONNX 모델을 INT8로 정적 양자화한다 (QDQ 형식, ONNX Runtime).

    uv run python scripts/quantize_onnx.py \
        --input models/model.onnx --output models/model.int8.onnx --calibration-dir data/calib

정적 양자화는 각 레이어 활성값의 범위를 정하려고 실제 이미지가 필요하다 (보정용 calibration 이미지).
학습 이미지 50~100장이면 충분하다. 전처리는 서빙과 같은 pill_serving.preprocess를 쓴다.

YOLO 출력은 박스 좌표(0~입력 크기)와 클래스 점수(0~1)를 한 텐서로 합치기 때문에,
마지막 노드까지 양자화하면 점수 정밀도가 크게 떨어질 수 있다.
benchmark.py로 FP32 모델과의 탐지 일치율을 확인하고, 낮으면 --exclude-nodes로 헤드 끝 노드를 빼고 다시 만든다
(노드 이름은 Netron으로 확인).
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from pill_serving.preprocess import load_image, preprocess

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class ImageFolderReader(CalibrationDataReader):
    """보정용 이미지를 하나씩 서빙과 같은 방식으로 전처리해서 넘긴다."""

    def __init__(self, paths: list[Path], input_name: str, input_size: int) -> None:
        self._items = iter(
            {input_name: preprocess(load_image(path.read_bytes()), input_size)} for path in paths
        )

    def get_next(self):
        return next(self._items, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="FP32 ONNX → INT8 정적 양자화 (QDQ)")
    parser.add_argument("--input", type=Path, required=True, help="FP32 ONNX 모델")
    parser.add_argument("--output", type=Path, required=True, help="INT8 ONNX 저장 경로")
    parser.add_argument("--calibration-dir", type=Path, required=True, help="보정용 이미지 폴더")
    parser.add_argument("--num-images", type=int, default=100, help="보정에 쓸 최대 이미지 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-nodes", nargs="*", default=None, help="양자화하지 않을 노드 이름")
    parser.add_argument("--per-tensor", action="store_true", help="채널별(per-channel) 대신 텐서 단위로 가중치 양자화")
    args = parser.parse_args()

    paths = sorted(p for p in args.calibration_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"보정용 이미지가 없습니다: {args.calibration_dir}")
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.num_images]

    model_input = ort.InferenceSession(str(args.input), providers=["CPUExecutionProvider"]).get_inputs()[0]
    input_size = int(model_input.shape[2])
    print(f"입력 {model_input.name} {model_input.shape}, 보정 이미지 {len(paths)}장")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        # shape 추론·그래프 정리 (양자화 권장 전처리). 입력 크기가 고정이라 symbolic shape 추론은 건너뛴다.
        quant_pre_process(str(args.input), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(args.output),
            ImageFolderReader(paths, model_input.name, input_size),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=not args.per_tensor,
            nodes_to_exclude=args.exclude_nodes or None,
        )

    before = args.input.stat().st_size / 1e6
    after = args.output.stat().st_size / 1e6
    print(f"저장: {args.output}  ({before:.1f} MB → {after:.1f} MB, {after / before:.0%})")


if __name__ == "__main__":
    main()
