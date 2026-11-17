"""Streamlit 웹 앱이 API 상태와 관계없이 예외 없이 그려지는지 확인한다 (streamlit.testing.AppTest)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "web" / "streamlit_app.py"


def test_renders_and_explains_when_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # 아무것도 듣지 않는 포트
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "💊 PillDetect 알약 탐지 데모"
    assert at.warning, "의료 목적 사용 금지 안내가 보여야 한다"
    assert any("연결할 수 없습니다" in e.value for e in at.sidebar.error)
