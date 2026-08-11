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

# 3. 指定された6項目の領域座標 (x1, y1, x2, y2)
CROP_AREAS = {
    "バットスピード": (0, 860, 355, 1010),
    "アッパー度": (355, 860, 710, 1010),
    "オンプレーン効率": (710, 860, 1067, 1010),
    "加速度": (0, 1200, 355, 1310),
    "スイング時間": (355, 1180, 710, 1300),
    "パワー": (710, 1080, 1067, 1300),
}


def fix_numeric_format(val_str, metric_name):
    """
    指定された数値ルールに基づいたフォーマット補正
    - バットスピード / 加速度: 小数点以下1桁
    - アッパー度 / オンプレーン効率: 整数
    - パワー: 小数点以下2桁
    - スイング時間: 読み取った数字を小数点以下にする (例: 22 -> 0.22)
    """
    if not val_str:
        return None

    # 数字のみ抽出（マイナス符号が必要な場合はマイナスも含める）
    digits_only = re.sub(r"[^\d-]", "", val_str)
    if not digits_only:
        return None

    try:
        # 1. スイング時間 (表示されている数字を0.XXとして出力)
        if metric_name == "スイング時間":
            num = int(re.sub(r"\D", "", val_str))
            return round(num / 100.0, 2)

        # 2. アッパー度 / オンプレーン効率 (整数)
        elif metric_name in ["アッパー度", "オンプレーン効率"]:
            return int(float(val_str))

        # 3. バットスピード / 加速度 (小数点以下1桁)
        elif metric_name in ["バットスピード", "加速度"]:
            # 小数点が含まれていない桁連続の場合の対応（例: 1355 -> 135.5）
            if "." not in val_str and len(digits_only) >= 2:
                val = float(f"{digits_only[:-1]}.{digits_only[-1]}")
            else:
                val = float(val_str)
            return round(val, 1)

        # 4. パワー (小数点以下2桁)
        elif metric_name == "パワー":
            if "." not in val_str and len(digits_only) >= 3:
                val = float(f"{digits_only[:-2]}.{digits_only[-2:]}")
            else:
                val = float(val_str)
            return round(val, 2)

        return float(val_str)

    except Exception:
        # 変換失敗時は元の文字列を返却
        return val_str


def process_image_fast(uploaded_file, index):
    """アップロードされた画像の読み取り処理"""
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
        # 拡大＆グレースケール化で認識精度向上
        large = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)

        # 数字・小数点・マイナス記号のみを対象にOCR実行
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

    # 並列処理で複数画像を同時解析
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

    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
