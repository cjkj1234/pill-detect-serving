"""클래스 매핑 로드 — PillDetect configs/class_mapping.json과 같은 형식."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PillClass:
    class_id: int      # 모델 출력 인덱스 (0 ~ nc-1)
    category_id: int   # Kaggle 제출용 원본 category_id
    name: str          # 약 이름


def load_class_mapping(path: str | Path) -> list[PillClass]:
    """class_id 순서로 정렬된 클래스 목록을 반환한다. class_id가 0부터 연속이 아니면 ValueError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"클래스 매핑 파일이 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    classes = sorted(
        (
            PillClass(
                class_id=int(item["class_id"]),
                category_id=int(item["category_id"]),
                name=str(item["name"]),
            )
            for item in data.get("classes", [])
        ),
        key=lambda c: c.class_id,
    )
    if not classes:
        raise ValueError(f"클래스 매핑에 classes가 없습니다: {path}")
    if [c.class_id for c in classes] != list(range(len(classes))):
        raise ValueError("class_id는 0부터 연속된 정수여야 합니다.")
    return classes
