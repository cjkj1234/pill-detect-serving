"""YOLO raw output 후처리 — NumPy 구현.

PillDetect src/engine/postprocess.py(Ultralytics non_max_suppression + class_probs)와 같은 규칙을
PyTorch 없이 구현했다. 서빙 이미지에서 torch·ultralytics를 빼서 이미지 크기와 시작 시간을 줄이기 위해서다.

규칙 (Ultralytics 8.4 non_max_suppression, multi_label=False 기준):
  1. 앵커마다 가장 높은 클래스 점수와 그 클래스를 고른다.
  2. 점수가 conf_threshold보다 큰 앵커만 남긴다.
  3. cx, cy, w, h → x1, y1, x2, y2로 바꾼다.
  4. class-aware NMS는 클래스마다 박스를 MAX_WH만큼 밀어서 다른 클래스끼리는 겹치지 않게 한다.
  5. 점수 내림차순 greedy NMS (IoU > iou_threshold면 제거) 후 max_detections개까지 남긴다.
  6. class_probs = 남은 박스가 나온 앵커의 클래스 점수를 합 1로 정규화한 값.

점수가 완전히 같은 박스는 앵커 번호가 작은 쪽을 먼저 둔다 (안정 정렬).
Ultralytics(torch argsort)는 동점 순서를 보장하지 않으므로 이 경우만 순서가 다를 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_WH = 7680      # Ultralytics와 같은 클래스별 NMS 오프셋
MAX_NMS = 30_000   # NMS에 넣는 최대 박스 수


@dataclass(frozen=True)
class PostprocessConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 4
    class_agnostic: bool = True


@dataclass
class RawDetections:
    """모델 입력 좌표계(정사각형 input_size) 기준 탐지 결과."""

    boxes: np.ndarray        # [N, 4] float32, x1 y1 x2 y2
    scores: np.ndarray       # [N] float32
    labels: np.ndarray       # [N] int64
    class_probs: np.ndarray  # [N, nc] float32, 행 합 1
    anchor_idxs: np.ndarray  # [N] int64, 박스가 나온 앵커 번호

    def __len__(self) -> int:
        return int(self.scores.shape[0])


def empty_detections(num_classes: int) -> RawDetections:
    return RawDetections(
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        class_probs=np.zeros((0, num_classes), dtype=np.float32),
        anchor_idxs=np.zeros((0,), dtype=np.int64),
    )


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[N, 4] cx, cy, w, h → x1, y1, x2, y2."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """점수 내림차순 greedy NMS. 남길 박스의 인덱스를 점수 내림차순으로 반환한다.

    torchvision.ops.nms와 같은 규칙: 이미 고른 박스와 IoU가 iou_threshold보다 크면 제거한다.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        inter_w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        inter_h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def postprocess(raw_output: np.ndarray, config: PostprocessConfig | None = None) -> RawDetections:
    """이미지 1장의 raw output [4 + nc, num_anchors] (또는 [1, 4 + nc, num_anchors])을 후처리한다.

    raw_output[4:]는 YOLO Detect head가 sigmoid를 적용한 0~1 클래스 점수다.
    """
    config = config or PostprocessConfig()

    if raw_output.ndim == 3:
        if raw_output.shape[0] != 1:
            raise ValueError("배치 크기 1인 raw output만 지원합니다.")
        raw_output = raw_output[0]
    if raw_output.ndim != 2 or raw_output.shape[0] <= 4:
        raise ValueError("raw output은 [4 + num_classes, num_anchors] 형태여야 합니다.")

    num_classes = raw_output.shape[0] - 4
    preds = np.asarray(raw_output, dtype=np.float32).T          # [A, 4 + nc]
    class_scores = preds[:, 4:]                                 # [A, nc]
    best_scores = class_scores.max(axis=1)
    best_labels = class_scores.argmax(axis=1)

    candidates = np.flatnonzero(best_scores > config.conf_threshold)
    if candidates.size == 0:
        return empty_detections(num_classes)
    if candidates.size > MAX_NMS:
        top = np.argsort(-best_scores[candidates], kind="stable")[:MAX_NMS]
        candidates = candidates[top]

    boxes = xywh_to_xyxy(preds[candidates, :4])
    scores = best_scores[candidates]
    labels = best_labels[candidates]

    nms_boxes = boxes if config.class_agnostic else boxes + labels[:, None].astype(np.float32) * MAX_WH
    keep = nms(nms_boxes, scores, config.iou_threshold)[: config.max_detections]

    anchor_idxs = candidates[keep]
    probs = class_scores[anchor_idxs]
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    return RawDetections(
        boxes=boxes[keep].astype(np.float32),
        scores=scores[keep].astype(np.float32),
        labels=labels[keep].astype(np.int64),
        class_probs=probs.astype(np.float32),
        anchor_idxs=anchor_idxs.astype(np.int64),
    )
