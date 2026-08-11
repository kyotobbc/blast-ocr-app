import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# 1. ページ基本設定
st.set_page_config(page_title="BLAST データ解析", layout="wide")
st.title("BLAST データ解析 (6項目抽出・微調整機能付)")
st.write("Blast Motion のスクショ画像をアップロードしてください。")

# 2. EasyOCRの初期化
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# ベースとなる元の座標 (x1, y1, x2, y2)
BASE_CROP_AREAS = {
    "バットスピード": (0, 860, 355, 1010),
    "アッパー度": (355, 860, 710, 1010),
    "オンプレーン効率": (710, 860, 1067, 1010),
    "加速度": (0, 1200, 355, 1310),
    "スイング時間": (355, 1180, 710, 1300),
    "パワー": (710, 1080, 1067, 1300),
}


def preprocess_for_ocr(crop_img):
    """OCR前処理"""
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    padded = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded


def fix_numeric_format(val_str, metric_name):
    """数値フォーマット補正"""
    if not val_str:
        return None

    digits_only = re.sub(r"[^\d-]", "", val_str)
    if not digits_only:
        return None

    try:
        if metric_name == "スイング時間":
            num = int(re.sub(r"\D", "", val_str))
            return round(num / 100.0, 2)

        elif metric_name in ["アッパー度", "オンプレーン効率"]:
            return int(float(val_str))

        elif metric_name in ["バットスピード", "加速度"]:
            if "." not in val_str and len(digits_only) >= 2:
                val = float(f"{digits_only[:-1]}.{digits_only[-1]}")
            else:
                val = float(val_str)
            return round(val, 1)

        elif metric_name == "パワー":
            if "." not in val_str and len(digits_only) >= 3:
                val = float(f"{digits_only[:-2]}.{digits_only[-2:]}")
            else:
                val = float(val_str)
            return round(val, 2)

        return float(val_str)

    except Exception:
        return val_str


def process_image(uploaded_file, offset_y=0, show_debug=False):
    """画像切り出し・読み取り処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)
    img_h, img_w = open_cv_full.shape[:2]

    results = {"ファイル名": uploaded_file.name}
    debug_images = {}

    for metric_name, (x1, y1, x2, y2) in BASE_CROP_AREAS.items():
        # オフセット適用（マイナスで上方向へ移動）
        ny1 = max(0, y1 + offset_y)
        ny2 = max(0, y2 + offset_y)
        nx1 = max(0, min(x1, img_w))
        nx2 = max(0, min(x2, img_w))

        # 画像範囲を超えないように安全制御
        ny1 = min(ny1, img_h)
        ny2 = min(ny2, img_h)

        region = open_cv_full[ny1:ny2, nx1:nx2]
        processed = preprocess_for_ocr(region)

        if processed is None:
            results[metric_name] = None
            continue

        if show_debug:
            debug_images[metric_name] = processed

        ocr_out = reader.readtext(
            processed,
            detail=0,
            allowlist="0123456789.-",
        )
        raw_ocr = "".join(ocr_out)

        match = re.search(r"-?\d+\.?\d*", raw_ocr)
        if match:
            results[metric_name] = fix_numeric_format(match.group(0), metric_name)
        else:
            results[metric_name] = None

    return results, debug_images


# --- サイドバー／画面設定 ---
st.sidebar.header("⚙️ 座標切り出し位置の微調整")
# デフォルトで「100px上に移動」に設定（スライダーで上下に動かせます）
y_offset = st.sidebar.slider(
    "縦方向の位置調整 (マイナスで上へ移動)",
    min_value=-500,
    max_value=200,
    value=-100,
    step=10
)

show_debug = st.sidebar.checkbox("【プレビュー表示】切り出し画像を画面で確認する", value=True)

uploaded_files = st.file_uploader(
    "BLASTのスクショ画像を選択（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("読み取る"):
    raw_results = []
    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files, 1):
        status_text.text(f"🚀 解析中 ({idx}/{total}): {file.name}")
        res, debug_imgs = process_image(file, offset_y=y_offset, show_debug=show_debug)
        raw_results.append(res)

        if show_debug and debug_imgs:
            st.write(f"--- プレビュー (Yオフセット: {y_offset}px): {file.name} ---")
            cols = st.columns(len(debug_imgs))
            for i, (k, img_crop) in enumerate(debug_imgs.items()):
                with cols[i]:
                    st.image(img_crop, caption=k, use_container_width=True)

        progress_bar.progress(idx / total)

    if raw_results:
        status_text.text("✅ 解析完了！")
        st.session_state["parsed_df"] = pd.DataFrame(raw_results)

if "parsed_df" in st.session_state:
    st.subheader("【解析結果】")
    st.data_editor(
        st.session_state["parsed_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
