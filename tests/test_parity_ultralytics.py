"""NumPy 후처리가 Ultralytics NMS(학습 저장소가 쓰는 구현)와 같은 결과를 내는지 확인한다.

torch·ultralytics가 설치돼 있을 때만 실행된다 (서빙 이미지와 기본 CI에는 넣지 않는다).

    uv run --with torch --with torchvision --with ultralytics==8.4.51 pytest tests/test_parity_ultralytics.py
"""

import numpy as np
import pytest

from pill_serving.postprocess import PostprocessConfig, postprocess

torch = pytest.importorskip("torch")
nms_module = pytest.importorskip("ultralytics.utils.nms")


def random_raw_output(seed: int, num_classes: int = 8, num_anchors: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 640, (2, num_anchors))
    sizes = rng.uniform(8, 120, (2, num_anchors))
    # 4제곱으로 대부분의 점수를 낮춰서 실제 모델처럼 소수의 앵커만 threshold를 넘게 한다
    scores = rng.uniform(0, 1, (num_classes, num_anchors)) ** 4
    return np.concatenate([centers, sizes, scores]).astype(np.float32)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("agnostic", [True, False])
@pytest.mark.parametrize("iou_threshold,max_det", [(0.7, 4), (0.5, 300)])
def test_matches_ultralytics_non_max_suppression(seed, agnostic, iou_threshold, max_det):
    raw = random_raw_output(seed)
    config = PostprocessConfig(
        conf_threshold=0.25, iou_threshold=iou_threshold, max_detections=max_det, class_agnostic=agnostic
    )

    ours = postprocess(raw, config)

    detections, kept = nms_module.non_max_suppression(
        torch.from_numpy(raw.copy())[None],
        conf_thres=config.conf_threshold,
        iou_thres=config.iou_threshold,
        agnostic=config.class_agnostic,
        max_det=config.max_detections,
        nc=raw.shape[0] - 4,
        return_idxs=True,
    )
    det = detections[0].numpy()
    kept_idxs = kept[0].long().numpy()

    assert len(ours) == det.shape[0] > 0

    # 점수가 float32로 완전히 같은 박스끼리는 순서가 구현마다 다르다
    # (torch argsort는 동점 순서를 보장하지 않고, NumPy 구현은 앵커 번호 순).
    # 그래서 양쪽 모두 (점수 내림차순, 앵커 번호 오름차순)으로 정렬한 뒤 비교한다.
    ours_order = np.lexsort((ours.anchor_idxs, -ours.scores))
    ref_order = np.lexsort((kept_idxs, -det[:, 4]))

    assert np.array_equal(ours.anchor_idxs[ours_order], kept_idxs[ref_order])
    assert np.allclose(ours.boxes[ours_order], det[ref_order, :4], atol=1e-3)
    assert np.allclose(ours.scores[ours_order], det[ref_order, 4], atol=1e-6)
    assert np.array_equal(ours.labels[ours_order], det[ref_order, 5].astype(np.int64))
