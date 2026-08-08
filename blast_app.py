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

# 3. 座標設定（時刻を一番最後/右側に配置）
NUMERIC_AREAS = {
    "バットスピード": (0, 855, 355, 995),
    "アッパー度": (355, 855, 710, 995),
    "オンプレーン効率": (710, 855, 1067, 995),
    "加速度": (0, 1185, 355, 1290),
    "スイング時間": (355, 1165, 710, 1285),
    "パワー": (710, 1165, 1067, 1285),
    "時刻": (525, 45, 960, 115),
}


def fix_numeric_format(val, metric_name):
    """桁数ルールおよび時刻専用（時:分:秒）抽出による精度向上ロジック"""
    if val is None:
        return None
    val_str = str(val).strip()

    if metric_name == "時刻":
        # OCR誤認識の文字補正 (O->0, I/l->1, S->5 などの代表的な置換)
        cleaned = val_str.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        
        # 時:分:秒（例: 8:32:05 または 15:48:31）のパターンのみを厳密に抽出
        match = re.search(r"(\d{1,2}:\d{2}:\d{2})", cleaned)
        if match:
            return match.group(1)
        
        # 万が一コロンがドットやカンマに誤認された場合のレスキュー処理
        alt_cleaned = re.sub(r"[.,;]", ":", cleaned)
        alt_match = re.search(r"(\d{1,2}:\d{2}:\d{2})", alt_cleaned)
        if alt_match:
            return alt_match.group(1)

        return None

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
    """精度補正を加えた画像解析処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)

    results = {"ファイル名": uploaded_file.name}

    for metric_name, (left, top, right, bottom) in NUMERIC_AREAS.items():
        region = open_cv_full[top:bottom, left:right]

        if region.size == 0:
            results[metric_name] = None
            continue

        h, w = region.shape[:2]

        if metric_name == "時刻":
            # 時刻領域は鮮明度を上げるため3倍にリサイズして二値化
            large = cv2.resize(region, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            raw_ocr = reader.readtext(
                thresh,
                detail=0,
                allowlist="0123456789:.,oOIilS ",
                paragraph=False,
            )
            combined_text = "".join(raw_ocr)
            final_val = fix_numeric_format(combined_text, metric_name)
        else:
            large = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)
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

            # 【1. Excelへ一括コピー（時刻は一番右端）】
            df_for_excel = df.drop(columns=["ファイル名"], errors="ignore")
            tsv_data = df_for_excel.to_csv(index=False, header=False, sep="\t")

            st.markdown("### 📋 Excelへ一括コピー")
            st.write(
                "下の枠内のデータ（数値データ＋時刻／ヘッダーなし）を全選択してコピー（Ctrl+C）し、Excelのセルにそのまま貼り付けてください（Ctrl+V）。"
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
