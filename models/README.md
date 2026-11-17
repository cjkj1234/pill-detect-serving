# models/

- `class_mapping.json` — 56개 알약 클래스 (`class_id` ↔ Kaggle `category_id` ↔ 약 이름). PillDetect `configs/class_mapping.json`과 같다.
- `model.onnx` — 서빙할 모델. **저장소에 올리지 않는다** (`.gitignore`).
  - 실제 모델: PillDetect `scripts/export_onnx.py`로 만든 파일을 복사한다.
  - 더미 모델: `uv run python scripts/make_dummy_model.py --output models/model.onnx`

다른 파일 이름을 쓰려면 `MODEL_PATH`(Docker Compose에서는 `MODEL_FILE`) 환경 변수로 지정한다.
