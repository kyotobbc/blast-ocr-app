import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# 1. ページの設定（タイトルの変更）
st.set_page_config(page_title="BLAST画像読み取り", layout="centered")

st.title("BLAST画像読み取り")
st.write(
    "スマホやPCのアルバムから Blast Motion のスクショ画像を複数選択してアップロードしてください。"
)

# 2. OCRエンジンの初期化（高速化設定）
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False, quantize=True)

reader = load_ocr()

# 3. 座標設定（時刻を追加）
NUMERIC_AREAS = {
    "時刻": (525, 45, 960, 115),
    "バットスピード": (0, 855, 355, 995),
    "アッパー度": (355, 855, 710, 995),
    "オンプレーン効率": (710, 855, 1067, 995),
    "加速度": (0, 1185, 355, 1290),
    "スイング時間": (355, 1165, 710, 1285),
    "パワー": (710, 1165, 1067, 1285),
}


def fix_numeric_format(val, metric_name):
    """桁数ルールに基づく自動補正"""
    if val is None:
        return None
    val_str = str(val).strip()

    if metric_name == "時刻":
        # 時刻用フォーマット調整（記号整形）
        cleaned = re.sub(r"[^\d:APM\s]", "", val_str, flags=re.IGNORECASE)
        return cleaned if cleaned else val_str

    elif metric_name == "加速度":
        digits_only = re.sub(r"\D", "", val_str)
        if len(digits_only) == 3:
            return float(f"{digits_only[:2]}.{digits_only[2:]}")
        elif len(digits_only) == 2:
            return float(f"{digits_only}.0")
        try:
            return round(float(val_str), 1)
        except ValueError:
            return val

    elif metric_name == "スイング時間":
        digits_only = re.sub(r"\D", "", val_str)
        if len(digits_only) >= 2:
            last_two = digits_only[-2:]
            return float(f"0.{last_two}")
        try:
            return round(float(val_str), 2)
        except ValueError:
            return val
    else:
        try:
            return float(val_str)
        except ValueError:
            return val


def process_image_fast(uploaded_file):
    """高速化版の画像解析処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)

    results = {"ファイル名": uploaded_file.name}

    for metric_name, (left, top, right, bottom) in NUMERIC_AREAS.items():
        region = open_cv_full[top:bottom, left:right]

        if region.size == 0:
            results[metric_name] = None
            continue

        h, w = region.shape[:2]
        large = cv2.resize(
            region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)

        # 時刻の場合は英字やコロンも許可する
        if metric_name == "時刻":
            raw_ocr = reader.readtext(
                gray,
                detail=0,
                allowlist="0123456789:APMapm ",
                paragraph=False,
                mag_ratio=1.0,
            )
            combined_text = " ".join(raw_ocr)
            final_val = fix_numeric_format(combined_text, metric_name)
        else:
            raw_ocr = reader.readtext(
                gray,
                detail=0,
                allowlist="0123456789.-",
                paragraph=False,
                mag_ratio=1.0,
            )
            combined_text = "".join(raw_ocr)
            match = re.search(r"-?\d+\.?\d*", combined_text)
            if match:
                raw_val = match.group(0)
                final_val = fix_numeric_format(raw_val, metric_name)
            else:
                final_val = None

        results[metric_name] = final_val

    return results


# --- 画面UI部分 ---
uploaded_files = st.file_uploader(
    "スクショ画像を選択（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("読み取る"):
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(uploaded_files)
        for i, file in enumerate(uploaded_files):
            status_text.text(f"解析中 ({i+1}/{total}): {file.name}")

            try:
                data = process_image_fast(file)
                all_data.append(data)
            except Exception as e:
                st.error(
                    f"ファイル '{file.name}' の処理中にエラーが発生しました: {e}"
                )

            progress_bar.progress((i + 1) / total)

        if all_data:
            status_text.text("✅ すべての画像の解析が完了しました！")
            df = pd.DataFrame(all_data)

            # 【1. Excelへ一括コピー（時刻含むデータ・ヘッダーなし）】
            df_for_excel = df.drop(columns=["ファイル名"], errors="ignore")
            tsv_data = df_for_excel.to_csv(index=False, header=False, sep="\t")

            st.markdown("### 📋 Excelへ一括コピー")
            st.write(
                "下の枠内のデータ（時刻＋数値データ／ヘッダーなし）を全選択してコピー（Ctrl+C）し、Excelのセルにそのまま貼り付けてください（Ctrl+V）。"
            )
            st.code(tsv_data, language="text")

            # 【2. 解析結果一覧（テーブル表示）】
            st.subheader("【解析結果一覧】")
            st.dataframe(df)

            # 【3. CSVダウンロードボタン】
            csv_data = df.to_csv(index=False, encoding="utf-8-sig").encode(
                "utf-8-sig"
            )
            st.download_button(
                label="📥 CSVファイルとして保存",
                data=csv_data,
                file_name="blast_extracted_data.csv",
                mime="text/csv",
            )
