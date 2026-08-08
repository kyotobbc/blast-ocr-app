import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# 1. ページの設定
st.set_page_config(page_title="BLAST画像読み取り", layout="wide")

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

# 4. 各指標の現実的な想定範囲（下限, 上限）
METRIC_RANGES = {
    "バットスピード": (30.0, 170.0),
    "アッパー度": (-30.0, 45.0),
    "オンプレーン効率": (20.0, 100.0),
    "加速度": (3.0, 35.0),
    "スイング時間": (0.08, 0.45),
    "パワー": (0.5, 12.0),
}


def fix_numeric_format(val, metric_name):
    """桁数ルールおよび時刻専用抽出ロジック"""
    if val is None:
        return None
    val_str = str(val).strip()

    if metric_name == "時刻":
        cleaned = val_str.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        match = re.search(r"(\d{1,2}:\d{2}:\d{2})", cleaned)
        if match:
            return match.group(1)
        
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
    """画像解析処理"""
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


def check_outliers(df):
    """標準偏差(3σ)および設定範囲チェックによる異常値フラグ設定"""
    status_list = []

    for idx, row in df.iterrows():
        reasons = []

        # 1. 範囲チェック & 2. 標準偏差チェック
        for metric, (min_val, max_val) in METRIC_RANGES.items():
            val = row.get(metric)
            if pd.notna(val) and isinstance(val, (int, float)):
                # 想定範囲外チェック
                if val < min_val or val > max_val:
                    reasons.append(f"{metric}範囲外({val})")

                # 標準偏差(3σ)チェック (データ数が4件以上ある場合)
                col_data = pd.to_numeric(df[metric], errors="coerce").dropna()
                if len(col_data) >= 4:
                    mean = col_data.mean()
                    std = col_data.std()
                    if std > 0 and abs(val - mean) > 3 * std:
                        reasons.append(f"{metric}外れ値({val})")

        if reasons:
            status_list.append("⚠️ " + ", ".join(reasons))
        else:
            status_list.append("✅ 正常")

    df.insert(1, "判定", status_list)
    return df


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
            
            # 異常値チェックを適用
            df = check_outliers(df)
            
            # 編集可能テーブル用にセッション状態へ保持
            st.session_state["parsed_df"] = df

if "parsed_df" in st.session_state:
    st.subheader("【解析結果一覧・手修正】")
    st.info("💡 画面上のセルを直接ダブルタップ/ダブルクリックして数値を変更・修正できます。誤検出行は左端を選択して削除できます。")

    # 手修正が可能なデータエディタを表示
    edited_df = st.data_editor(
        st.session_state["parsed_df"],
        num_rows="dynamic", # 行の削除や追加を許可
        use_container_width=True,
        key="data_editor"
    )

    # 画面で修正された最新のデータを反映
    st.markdown("### 📋 Excelへ一括コピー")
    st.write(
        "下の枠内のデータ（数値データ＋時刻／ヘッダーなし）を全選択してコピー（Ctrl+C）し、Excelのセルにそのまま貼り付けてください（Ctrl+V）。"
    )

    # Excel出力用（「ファイル名」と「判定」列を除外）
    export_df = edited_df.drop(columns=["ファイル名", "判定"], errors="ignore")
    tsv_data = export_df.to_csv(index=False, header=False, sep="\t")
    
    st.code(tsv_data, language="text")

    # CSVダウンロードボタン
    csv_data = edited_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 CSVファイルとして保存",
        data=csv_data,
        file_name="blast_extracted_data.csv",
        mime="text/csv",
    )
