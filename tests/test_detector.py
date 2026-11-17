import numpy as np
import pytest

from pill_serving.classes import PillClass, load_class_mapping
from pill_serving.detector import Detector
from pill_serving.preprocess import InvalidImageError, load_image, preprocess


@pytest.fixture(scope="module")
def detector(model_dir) -> Detector:
    return Detector(model_dir / "model.onnx", load_class_mapping(model_dir / "class_mapping.json"))


def test_reads_input_size_from_model(detector):
    assert detector.input_size == 640
    assert detector.providers == ["CPUExecutionProvider"]


def test_predict_scales_boxes_back_to_original_image(detector, png_bytes):
    image = load_image(png_bytes)  # 976 × 1280 (W × H)
    result = detector.predict(image)

    assert (result.width, result.height) == (976, 1280)
    assert [d.pill.class_id for d in result.detections] == [0, 2, 0]
    # 박스 A는 640 입력에서 (40, 40, 60, 60) → x ×976/640, y ×1280/640
    assert np.allclose(result.detections[0].box, (61.0, 80.0, 91.5, 120.0))
    assert set(result.timing_ms) == {"preprocess", "inference", "postprocess", "total"}


def test_candidates_are_sorted_and_start_with_the_label(detector, png_bytes):
    result = detector.predict(load_image(png_bytes))

    box_b = result.detections[2]  # 클래스 0 (0.60), 2위 클래스 1 (0.35)
    assert [c.pill.class_id for c in box_b.candidates] == [0, 1, 2]
    probs = [c.prob for c in box_b.candidates]
    assert probs == sorted(probs, reverse=True)
    assert np.isclose(sum(probs), 1.0, atol=1e-5)
    assert np.isclose(probs[0], 0.60, atol=1e-5)


def test_rejects_class_mapping_with_wrong_size(model_dir):
    wrong = [PillClass(class_id=i, category_id=i, name=str(i)) for i in range(5)]
    with pytest.raises(ValueError, match="클래스 수"):
        Detector(model_dir / "model.onnx", wrong)


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector(tmp_path / "nope.onnx", [])


def test_preprocess_shape_and_range():
    image = np.full((1280, 976, 3), 255, dtype=np.uint8)
    x = preprocess(image, 640)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32
    assert np.allclose(x, 1.0)


def test_load_image_rejects_non_image():
    with pytest.raises(InvalidImageError):
        load_image(b"not an image")
