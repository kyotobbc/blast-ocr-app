import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# 1. ページ基本設定
st.set_page_config(page_title="BLAST データ解析", layout="wide")
st.title("BLAST データ解析")
st.write("Blast Motion のスクショ画像をアップロードしてください。")

# 2. EasyOCRの初期化
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# 各項目の基本座標 (x1, y1, x2, y2)
BASE_CROP_AREAS = {
    "バットスピード": (0, 860, 355, 1010),
    "アッパー度": (355, 860, 710, 1010),
    "オンプレーン効率": (710, 860, 1067, 1010),
    "加速度": (0, 1200, 355, 1310),
    "スイング時間": (355, 1180, 710, 1300),
    "パワー": (710, 1080, 1067, 1300),
}


def preprocess_for_ocr(crop_img):
    """OCR精度向上のための前処理"""
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
    """各項目の表示ルールに合わせたフォーマット整形"""
    if not val_str:
        return None

    digits_only = re.sub(r"[^\d-]", "", val_str)
    if not digits_only:
        return None

    try:
        # スイング時間: 読み取った数字を 0.XX に変換
        if metric_name == "スイング時間":
            num = int(re.sub(r"\D", "", val_str))
            return round(num / 100.0, 2)

        # アッパー度 / オンプレーン効率: 整数
        elif metric_name in ["アッパー度", "オンプレーン効率"]:
            return int(float(val_str))

        # バットスピード / 加速度: 小数点以下1桁
        elif metric_name in ["バットスピード", "加速度"]:
            if "." not in val_str and len(digits_only) >= 2:
                val = float(f"{digits_only[:-1]}.{digits_only[-1]}")
            else:
                val = float(val_str)
            return round(val, 1)

        # パワー: 小数点以下2桁
        elif metric_name == "パワー":
            if "." not in val_str and len(digits_only) >= 3:
                val = float(f"{digits_only[:-2]}.{digits_only[-2:]}")
            else:
                val = float(val_str)
            return round(val, 2)

        return float(val_str)

    except Exception:
        return val_str


def process_image(uploaded_file):
    """画像から6項目を抽出し数値化する処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)
    img_h, img_w = open_cv_full.shape[:2]

    results = {"ファイル名": uploaded_file.name}
    offset_y = -20  # 最適化済みのYオフセット

    for metric_name, (x1, y1, x2, y2) in BASE_CROP_AREAS.items():
        ny1 = y1 + offset_y
        ny2 = y2 + offset_y

        # パワーは数値部分（下側 2/3）のみ抽出
        if metric_name == "パワー":
            box_height = ny2 - ny1
            ny1 = int(ny1 + (box_height * (1 / 3)))

        ny1 = max(0, min(ny1, img_h))
        ny2 = max(0, min(ny2, img_h))
        nx1 = max(0, min(x1, img_w))
        nx2 = max(0, min(x2, img_w))

        region = open_cv_full[ny1:ny2, nx1:nx2]
        processed = preprocess_for_ocr(region)

        if processed is None:
            results[metric_name] = None
            continue

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

    return results


# --- 画面UI ---
uploaded_files = st.file_uploader(
    "画像ファイルを選択（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("読み取る"):
    raw_results = []
    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files, 1):
        status_text.text(f"解析中... ({idx}/{total})")
        res = process_image(file)
        raw_results.append(res)
        progress_bar.progress(idx / total)

    status_text.text("完了！")
    st.session_state["parsed_df"] = pd.DataFrame(raw_results)

# 結果の表表示
if "parsed_df" in st.session_state:
    st.subheader("【解析結果】")
    st.data_editor(
        st.session_state["parsed_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
