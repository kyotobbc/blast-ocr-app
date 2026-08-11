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
st.title("BLAST データ解析 (6項目抽出)")
st.write("Blast Motion のスクショ画像をアップロードしてください。")

# 2. EasyOCRの初期化（キャッシュ化して高速化）
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False, quantize=True)

reader = load_ocr()

# 3. 指定の6項目と左上・右下座標 (x1, y1, x2, y2)
CROP_AREAS = {
    "バットスピード": (0, 855, 355, 995),
    "アッパー度": (355, 855, 710, 995),
    "オンプレーン効率": (710, 855, 1067, 995),
    "加速度": (0, 1185, 355, 1290),
    "スイング時間": (355, 1165, 710, 1285),
    "パワー": (710, 1165, 1067, 1285),
}

def fix_numeric_format(val_str, metric_name):
    """数値のフォーマット整形（桁数・小数点補正）"""
    if not val_str:
        return None
    
    digits_only = re.sub(r"\D", "", val_str)
    
    try:
        if metric_name == "加速度":
            if len(digits_only) == 3:
                return float(f"{digits_only[:2]}.{digits_only[2:]}")
            elif len(digits_only) == 2:
                return float(f"{digits_only}.0")
            return round(float(val_str), 1)
        elif metric_name == "スイング時間":
            if len(digits_only) >= 2:
                return float(f"0.{digits_only[-2:]}")
            return round(float(val_str), 2)
        return float(val_str)
    except ValueError:
        return val_str


def process_image_fast(uploaded_file, index):
    """アップロードされた画像の読み取り処理"""
    # StreamlitのファイルオブジェクトをOpenCV形式に変換
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)

    results = {"_index": index, "ファイル名": uploaded_file.name}

    for metric_name, (x1, y1, x2, y2) in CROP_AREAS.items():
        # 指定領域の切り出し [y1:y2, x1:x2]
        region = open_cv_full[y1:y2, x1:x2]

        if region.size == 0:
            results[metric_name] = None
            continue

        h, w = region.shape[:2]
        large = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)

        # OCR実行（数字と小数点・マイナスのみ認識）
        raw_ocr = "".join(
            reader.readtext(
                gray,
                detail=0,
                allowlist="0123456789.-",
                mag_ratio=1.0,
            )
        )
        
        match = re.search(r"-?\d+\.?\d*", raw_ocr)
        if match:
            results[metric_name] = fix_numeric_format(match.group(0), metric_name)
        else:
            results[metric_name] = None

    return results


# --- 画面UI・ファイルアップロード ---
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

    max_workers = min(4, total)
    status_text.text(f"🚀 解析中 ({total} 枚)...")

    # 並列処理で画像解析を高速化
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image_fast, file, idx): file
            for idx, file in enumerate(uploaded_files)
        }

        for completed_count, future in enumerate(as_completed(futures), 1):
            try:
                raw_results.append(future.result())
            except Exception as e:
                st.error(f"エラー ({futures[future].name}): {e}")

            progress_bar.progress(completed_count / total)

    if raw_results:
        status_text.text("✅ 解析完了！")
        # アップロード順にソートして一時キー削除
        raw_results.sort(key=lambda x: x["_index"])
        for r in raw_results:
            del r["_index"]

        df = pd.DataFrame(raw_results)
        st.session_state["parsed_df"] = df

# 結果表示枠
if "parsed_df" in st.session_state:
    current_df = st.session_state["parsed_df"]
    
    st.subheader("【解析結果】")
    st.info("💡 必要に応じて表内の数値を直接修正・編集できます。")
    
    # 編集可能な表で結果を表示
    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
