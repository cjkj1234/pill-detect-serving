"""ONNX 모델 여러 개의 크기·CPU 지연시간·탐지 일치율을 비교한다.

    uv run python scripts/benchmark.py models/model.onnx models/model.int8.onnx --images data/calib

- 첫 번째 모델을 기준으로 삼아, 나머지 모델이 같은 이미지에서 같은 알약을 찾는지 비교한다
  (같은 클래스이고 IoU ≥ 0.5인 박스를 짝지은 비율).
- 정확한 mAP는 정답 라벨이 필요하므로 학습 저장소(PillDetect)의 평가 코드로 따로 잰다.
- 결과는 Markdown 표로 출력한다 (--output을 주면 파일로도 저장).
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pill_serving.classes import load_class_mapping
from pill_serving.detector import Detector, PredictionResult
from pill_serving.preprocess import load_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Row:
    name: str
    size_mb: float
    inference_ms: list[float]
    total_ms: list[float]
    agreement: float | None
    score_diff: float | None


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference: list[PredictionResult], other: list[PredictionResult]) -> tuple[float, float | None]:
    """기준 탐지 중 같은 클래스·IoU ≥ 0.5로 짝지어진 비율과, 짝지어진 박스의 평균 점수 차."""
    matched, total, diffs = 0, 0, []
    for ref, oth in zip(reference, other):
        used: set[int] = set()
        for r in ref.detections:
            total += 1
            for j, o in enumerate(oth.detections):
                if j not in used and o.pill.class_id == r.pill.class_id and iou(r.box, o.box) >= 0.5:
                    used.add(j)
                    matched += 1
                    diffs.append(abs(o.score - r.score))
                    break
    agreement = matched / total if total else 1.0
    return agreement, (float(np.mean(diffs)) if diffs else None)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 모델 크기·지연시간·일치율 비교")
    parser.add_argument("models", nargs="+", type=Path, help="비교할 ONNX 모델 (첫 번째가 기준)")
    parser.add_argument("--images", type=Path, default=None, help="이미지 폴더 (없으면 무작위 이미지)")
    parser.add_argument("--class-mapping", type=Path, default=Path("models/class_mapping.json"))
    parser.add_argument("--runs", type=int, default=50, help="모델당 측정 횟수")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op 스레드 수 (0=기본)")
    parser.add_argument("--output", type=Path, default=None, help="Markdown 표 저장 경로")
    args = parser.parse_args()

    classes = load_class_mapping(args.class_mapping)
    if args.images:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        images = [load_image(p.read_bytes()) for p in paths]
        if not images:
            raise SystemExit(f"이미지가 없습니다: {args.images}")
    else:
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, (1280, 976, 3), dtype=np.uint8) for _ in range(5)]

    rows: list[Row] = []
    reference_results: list[PredictionResult] | None = None
    for model_path in args.models:
        detector = Detector(model_path, classes, intra_op_threads=args.threads)
        for i in range(args.warmup):
            detector.predict(images[i % len(images)])

        timings = [detector.predict(images[i % len(images)]).timing_ms for i in range(args.runs)]
        results = [detector.predict(image) for image in images]

        if reference_results is None:
            reference_results = results
            agreement, score_diff = None, None
        else:
            agreement, score_diff = compare(reference_results, results)

        rows.append(
            Row(
                name=model_path.name,
                size_mb=model_path.stat().st_size / 1e6,
                inference_ms=[t["inference"] for t in timings],
                total_ms=[t["total"] for t in timings],
                agreement=agreement,
                score_diff=score_diff,
            )
        )
        print(f"측정 완료: {model_path.name}")

    lines = [
        f"이미지 {len(images)}장 · 모델당 {args.runs}회 · CPU",
        "",
        "| 모델 | 크기 (MB) | 추론 p50 (ms) | 추론 p95 (ms) | 전체 평균 (ms) | 기준 대비 탐지 일치 | 평균 점수 차 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        agreement = "기준" if row.agreement is None else f"{row.agreement:.1%}"
        diff = "-" if row.score_diff is None else f"{row.score_diff:.3f}"
        lines.append(
            f"| {row.name} | {row.size_mb:.1f} | {percentile(row.inference_ms, 50):.1f} | "
            f"{percentile(row.inference_ms, 95):.1f} | {statistics.fmean(row.total_ms):.1f} | {agreement} | {diff} |"
        )
    table = "\n".join(lines)
    print()
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
