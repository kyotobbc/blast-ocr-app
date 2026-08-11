import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# 1. ページ基本設定
st.set_page_config(page_title="BLAST データ解析", layout="wide")
st.title("BLAST データ解析 (6項目抽出・自動スケール調整対応)")
st.write("Blast Motion のスクショ画像をアップロードしてください。（複数枚一括処理対応）")

# 2. EasyOCRの初期化
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_ocr()

# 3. 基準画像サイズ（基準座標を定義した際の画像解像度: 1067 x 1300 付近）
BASE_WIDTH = 1067
BASE_HEIGHT = 1300

# 基準座標 (x1, y1, x2, y2)
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
    """
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # 二値化処理
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 背景が黒・文字が白の場合は反転（白背景・黒文字へ統一）
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # 外枠余白（パディング）追加
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


def process_image(uploaded_file, show_debug=False):
    """画像読み取り・解析メイン処理（画像サイズに応じた自動スケール補正付）"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)

    img_h, img_w = open_cv_full.shape[:2]
    
    # 基準サイズとの比率計算（解像度が異なるスマホ画像でも座標を自動追従）
    scale_x = img_w / BASE_WIDTH
    scale_y = img_h / BASE_HEIGHT

    results = {"ファイル名": uploaded_file.name}
    debug_images = {}

    for metric_name, (x1, y1, x2, y2) in CROP_AREAS.items():
        # 画像解像度に合わせて座標をスケーリング
        sx1 = int(x1 * scale_x)
        sy1 = int(y1 * scale_y)
        sx2 = int(x2 * scale_x)
        sy2 = int(y2 * scale_y)

        # 画像切り出し [y1:y2, x1:x2]
        region = open_cv_full[sy1:sy2, sx1:sx2]

        processed = preprocess_for_ocr(region)

        if processed is None:
            results[metric_name] = None
            continue

        if show_debug:
            debug_images[metric_name] = processed

        # OCR実行
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

show_debug = st.checkbox("【確認用】切り出し後のプレビュー画像（デバッグ画像）を表示する")

if uploaded_files and st.button("読み取る"):
    raw_results = []
    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files, 1):
        status_text.text(f"🚀 解析中 ({idx}/{total}): {file.name}")
        res, debug_imgs = process_image(file, show_debug=show_debug)
        raw_results.append(res)

        # プレビュー表示（use_container_width=True に修正済み）
        if show_debug and debug_imgs:
            st.write(f"--- プレビュー: {file.name} (元サイズ: {Image.open(file).size[0]}x{Image.open(file).size[1]}px) ---")
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
