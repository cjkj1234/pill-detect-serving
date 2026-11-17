import numpy as np
import pytest

from pill_serving.postprocess import PostprocessConfig, nms, postprocess


def expected_probs(raw: np.ndarray, anchor: int) -> np.ndarray:
    scores = raw[4:, anchor]
    return scores / scores.sum()


def test_keeps_three_boxes_sorted_by_score(raw_output):
    result = postprocess(raw_output, PostprocessConfig(conf_threshold=0.25, iou_threshold=0.7))

    assert len(result) == 3
    # 앵커 1(박스 A와 겹침)은 NMS로, 앵커 4(낮은 점수)는 threshold로 빠진다.
    assert result.anchor_idxs.tolist() == [0, 3, 2]
    assert np.allclose(result.scores, [0.90, 0.70, 0.60])
    assert result.labels.tolist() == [0, 2, 0]
    # 앵커 0: cx 50, cy 50, w 20, h 20 → xyxy
    assert np.allclose(result.boxes[0], [40, 40, 60, 60])


def test_same_class_boxes_get_their_own_anchor_probs(raw_output):
    """같은 클래스 박스 A(0.90)와 B(0.60)가 각자 자기 앵커의 분포를 받는다."""
    result = postprocess(raw_output)

    for anchor, probs in zip(result.anchor_idxs, result.class_probs):
        assert np.allclose(probs, expected_probs(raw_output, anchor), atol=1e-6)


def test_class_probs_sum_to_one_and_argmax_matches_label():
    rng = np.random.default_rng(0)
    num_classes, num_anchors = 10, 500
    raw = np.concatenate(
        [
            rng.uniform(0, 640, (2, num_anchors)),
            rng.uniform(10, 100, (2, num_anchors)),
            rng.uniform(0, 1, (num_classes, num_anchors)),
        ]
    ).astype(np.float32)

    result = postprocess(raw, PostprocessConfig(conf_threshold=0.9, max_detections=50))

    assert len(result) > 0
    assert np.allclose(result.class_probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(result.class_probs.argmax(axis=1), result.labels)


def make_two_overlapping_boxes_of_different_classes() -> np.ndarray:
    # 같은 위치, 클래스 0(0.9)과 클래스 1(0.8)
    boxes = np.array([[100, 100, 40, 40], [101, 100, 40, 40]], dtype=np.float32)
    scores = np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)
    return np.concatenate([boxes, scores], axis=1).T


def test_class_agnostic_nms_suppresses_across_classes():
    raw = make_two_overlapping_boxes_of_different_classes()
    agnostic = postprocess(raw, PostprocessConfig(class_agnostic=True))
    aware = postprocess(raw, PostprocessConfig(class_agnostic=False))

    assert agnostic.labels.tolist() == [0]
    assert sorted(aware.labels.tolist()) == [0, 1]


def test_max_detections_limits_output(raw_output):
    result = postprocess(raw_output, PostprocessConfig(max_detections=2))
    assert result.anchor_idxs.tolist() == [0, 3]


def test_empty_result_keeps_shapes(raw_output):
    result = postprocess(raw_output, PostprocessConfig(conf_threshold=0.95))
    assert len(result) == 0
    assert result.boxes.shape == (0, 4)
    assert result.class_probs.shape == (0, raw_output.shape[0] - 4)


def test_accepts_batch_dimension(raw_output):
    a = postprocess(raw_output)
    b = postprocess(raw_output[None])
    assert np.array_equal(a.anchor_idxs, b.anchor_idxs)


@pytest.mark.parametrize("shape", [(4, 10), (2, 7, 10), (7,)])
def test_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        postprocess(np.zeros(shape, dtype=np.float32))


def test_nms_removes_only_boxes_above_iou_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    # 박스 0과 1의 IoU = 0.9
    assert nms(boxes, scores, 0.95).tolist() == [0, 1, 2]
    assert nms(boxes, scores, 0.85).tolist() == [0, 2]
