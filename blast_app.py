import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. ページ基本設定
st.set_page_config(page_title="BLAST データ解析", layout="wide")
st.title("BLAST データ解析 (6項目抽出・高精度版)")
st.write("Blast Motion のスクショ画像をアップロードしてください。")

# 2. EasyOCRの初期化
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# 3. 指定された6項目の領域座標 (x1, y1, x2, y2)
CROP_AREAS = {
    "バットスピード": (0, 860, 355, 1010),
    "アッパー度": (355, 860, 710, 1010),
    "オンプレーン効率": (710, 860, 1067, 1010),
    "加速度": (0, 1200, 355, 1310),
    "スイング時間": (355, 1180, 710, 1300),
    "パワー": (710, 1080, 1067, 1300),
}

def preprocess_for_ocr(crop_img):
    """
    OCR精度向上のための画像前処理
    - 拡大
    - グレースケール化
    - 二値化 (文字と背景の明確化)
    - 白黒反転（背景を白、文字を黒に統一）
    - 余白パディング（認識しやすくする）
    """
    if crop_img.size == 0:
        return None

    # グレースケール化
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    # 3倍に拡大
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # アダプティブ二値化または大津の二値化
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 文字が「白」背景が「黒」の場合は「黒文字・白背景」に反転（OCRは黒文字白背景が得意）
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # 外枠に余白（パディング）を追加して文字枠の端切れを防止
    padded = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded


def fix_numeric_format(val_str, metric_name):
    """数値フォーマット補正処理"""
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


def process_image_fast(uploaded_file, index, show_debug=False):
    """画像読み取り・解析メイン処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)

    results = {"_index": index, "ファイル名": uploaded_file.name}
    debug_images = {}

    for metric_name, (x1, y1, x2, y2) in CROP_AREAS.items():
        # 画像スライス [y1:y2, x1:x2]
        region = open_cv_full[y1:y2, x1:x2]

        # 前処理済みの画像作成
        processed = preprocess_for_ocr(region)

        if processed is None:
            results[metric_name] = None
            continue

        if show_debug:
            debug_images[metric_name] = processed

        # OCR実行 (前処理後の画像を使用)
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


# --- 画面UI ---
uploaded_files = st.file_uploader(
    "BLASTのスクショ画像を選択（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# デバッグ表示用チェックボックス
show_debug = st.checkbox("【確認用】切り出し後のプレビュー画像（デバッグ画像）を表示する")

if uploaded_files and st.button("読み取る"):
    raw_results = []
    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files, 1):
        status_text.text(f"🚀 解析中 ({idx}/{total}): {file.name}")
        res, debug_imgs = process_image_fast(file, idx - 1, show_debug=show_debug)
        raw_results.append(res)

        # 切り出し画像のデバッグ確認表示
        if show_debug and debug_imgs:
            st.write(f"--- デバッグ用切り出しプレビュー: {file.name} ---")
            cols = st.columns(len(debug_imgs))
            for i, (k, img_crop) in enumerate(debug_imgs.items()):
                with cols[i]:
                    st.image(img_crop, caption=k, use_column_width=True)

        progress_bar.progress(idx / total)

    if raw_results:
        status_text.text("✅ 解析完了！")
        df = pd.DataFrame(raw_results)
        if "_index" in df.columns:
            df = df.drop(columns=["_index"])
        st.session_state["parsed_df"] = df

if "parsed_df" in st.session_state:
    st.subheader("【解析結果】")
    st.data_editor(
        st.session_state["parsed_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
