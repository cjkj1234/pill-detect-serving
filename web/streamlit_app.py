"""PillDetect 데모 웹 — FastAPI 서빙 API를 호출해서 결과를 보여준다.

    API_URL=http://localhost:8000 streamlit run web/streamlit_app.py

이 앱은 모델을 직접 불러오지 않는다. 서빙 의존성(onnxruntime 등) 없이 requests·Pillow·Streamlit만 쓴다.
"""

from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DISPLAY_MAX_WIDTH = 720
COLORS = ["#E4572E", "#17BEBB", "#FFC914", "#76B041", "#9B5DE5", "#2E86AB"]

st.set_page_config(page_title="PillDetect 알약 탐지 데모", page_icon="💊", layout="wide")


@st.cache_data(ttl=10, show_spinner=False)
def fetch_json(path: str) -> tuple[int, dict]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=5)
        return response.status_code, response.json()
    except requests.RequestException as exc:
        return 0, {"detail": f"API에 연결할 수 없습니다: {exc}"}


def call_predict(name: str, data: bytes, content_type: str) -> tuple[int, dict]:
    try:
        response = requests.post(
            f"{API_URL}/predict", files={"file": (name, data, content_type)}, timeout=60
        )
        return response.status_code, response.json()
    except requests.RequestException as exc:
        return 0, {"detail": f"API에 연결할 수 없습니다: {exc}"}
    except ValueError:
        return 0, {"detail": "API 응답을 해석할 수 없습니다."}


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """박스와 번호만 그린다. 약 이름은 옆 표에 보여준다 (한글 폰트가 없는 환경에서도 깨지지 않게)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(min(canvas.size) / 200))
    font = ImageFont.load_default(size=max(18, round(min(canvas.size) / 25)))
    for number, det in enumerate(detections, start=1):
        color = COLORS[(number - 1) % len(COLORS)]
        box = det["box"]
        xy = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rectangle(xy, outline=color, width=line_width)
        label = str(number)
        text_box = draw.textbbox((xy[0], xy[1]), label, font=font)
        pad = line_width
        tag = (text_box[0], text_box[1] - (text_box[3] - text_box[1]) - 2 * pad, text_box[2] + 2 * pad, text_box[1])
        draw.rectangle(tag, fill=color)
        draw.text((tag[0] + pad, tag[1] + pad), label, fill="white", font=font)
    return canvas


def fit_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    height = round(image.height * max_width / image.width)
    return image.resize((max_width, height))


# --- 사이드바: API 상태 -----------------------------------------------------------
with st.sidebar:
    st.header("서빙 상태")
    st.caption(f"API: `{API_URL}`")
    status, health = fetch_json("/health")
    if status == 200 and health.get("model_loaded"):
        st.success("모델 로드됨")
        _, info = fetch_json("/model")
        st.markdown(
            f"- 모델 파일: `{info.get('model_file')}`\n"
            f"- 입력 크기: {info.get('input_size')} px\n"
            f"- 클래스 수: {info.get('num_classes')}\n"
            f"- 실행 장치: {', '.join(info.get('providers', []))}\n"
            f"- conf / IoU: {info.get('conf_threshold')} / {info.get('iou_threshold')}\n"
            f"- 이미지당 최대: {info.get('max_detections')}개"
        )
    elif status == 200:
        st.error("API는 실행 중이지만 모델이 로드되지 않았습니다.")
        st.caption(health.get("detail") or "")
    else:
        st.error(health.get("detail", "API 상태를 확인할 수 없습니다."))
    if st.button("상태 새로고침"):
        fetch_json.clear()
        st.rerun()

# --- 본문 -------------------------------------------------------------------------
st.title("💊 PillDetect 알약 탐지 데모")
st.write(
    "알약 사진을 올리면 최대 4개 알약의 위치와 이름을 찾습니다. "
    "모델은 YOLO11을 ONNX로 변환해 CPU에서 실행합니다."
)
st.warning(
    "연구·학습용 데모입니다. 실험실에서 촬영한 알약 사진으로 학습해서 일반 사진에서는 정확도가 낮습니다. "
    "복약 판단이나 약 식별 같은 의료 목적으로 사용하지 마세요.",
    icon="⚠️",
)

uploaded = st.file_uploader("알약 사진 (PNG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    data = uploaded.getvalue()
    try:
        original = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    except Exception:
        st.error("이미지를 열 수 없습니다. 다른 파일을 올려 주세요.")
        st.stop()

    with st.spinner("탐지 중..."):
        status, result = call_predict(uploaded.name, data, uploaded.type or "application/octet-stream")

    if status != 200:
        st.error(f"요청 실패 ({status or '연결 오류'}): {result.get('detail', result)}")
        st.stop()

    detections = result["detections"]
    left, right = st.columns([3, 2])

    with left:
        annotated = draw_boxes(original, detections)
        st.image(fit_width(annotated, DISPLAY_MAX_WIDTH), caption=f"{len(detections)}개 탐지")

    with right:
        if not detections:
            st.info("신뢰도 기준을 넘는 알약을 찾지 못했습니다.")
        else:
            rows = []
            for number, det in enumerate(detections, start=1):
                others = ", ".join(f"{c['name']} ({c['prob']:.0%})" for c in det["candidates"][1:])
                rows.append(
                    {
                        "번호": number,
                        "약 이름": det["name"],
                        "신뢰도": f"{det['score']:.1%}",
                        "다른 후보": others or "-",
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")

        timing = result["timing"]
        c1, c2, c3 = st.columns(3)
        c1.metric("전체", f"{timing['total_ms']:.0f} ms")
        c2.metric("추론", f"{timing['inference_ms']:.0f} ms")
        c3.metric("전·후처리", f"{timing['preprocess_ms'] + timing['postprocess_ms']:.0f} ms")

        with st.expander("API 응답 원본 (JSON)"):
            st.json(result)
