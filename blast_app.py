import gc
import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# 1. ページ基本設定
st.set_page_config(page_title="BLAST データ解析", layout="wide")
st.title("BLAST データ解析")
st.write("Blast Motion のスクショ画像をアップロードしてください。")

# 2. EasyOCRの初期化（軽量化設定）
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False, quantize=True)

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
    """OCR前処理（メモリ消費を極力抑える設定）"""
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded


def fix_numeric_format(val_str, metric_name):
    """各項目のフォーマット補正"""
    if not val_str:
        return ""

    digits_only = re.sub(r"[^\d-]", "", val_str)
    if not digits_only:
        return ""

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


def process_image(uploaded_file):
    """1枚ずつ処理し、メモリを極力開放する"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)
    img_h, img_w = open_cv_full.shape[:2]

    results = {}
    offset_y = -20

    for metric_name, (x1, y1, x2, y2) in BASE_CROP_AREAS.items():
        ny1 = y1 + offset_y
        ny2 = y2 + offset_y

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
            results[metric_name] = ""
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
            results[metric_name] = ""

    # 大きな画像データを明示的に解放
    del open_cv_full
    del img
    gc.collect()

    return results


# --- 画面UI ---
uploaded_files = st.file_uploader(
    "画像ファイルを選択（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("読み取る"):
    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    raw_results = []

    # メモリ超過を防ぐため、並列化せず1枚ずつ順番に処理
    for idx, file in enumerate(uploaded_files, 1):
        status_text.text(f"解析中... ({idx}/{total})")
        res = process_image(file)
        raw_results.append(res)
        progress_bar.progress(idx / total)

    status_text.text("完了！")

    # 6項目の値を「タブ（\t）」区切り文字列として整形
    metric_keys = ["バットスピード", "アッパー度", "オンプレーン効率", "加速度", "スイング時間", "パワー"]
    tsv_lines = []
    csv_lines = []
    for item in raw_results:
        row_values = [str(item.get(k, "")) for k in metric_keys]
        tsv_lines.append("\t".join(row_values))
        csv_lines.append(",".join(row_values))

    st.session_state["raw_tsv_text"] = "\n".join(tsv_lines)
    st.session_state["raw_csv_text"] = "\n".join(csv_lines)

# 結果表示部
if "raw_tsv_text" in st.session_state:
    st.subheader("【iPad対応 結果出力】")

    tsv_data = st.session_state["raw_tsv_text"]
    js_escaped_tsv = tsv_data.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    html_code = f"""
    <div style="margin-bottom: 15px;">
        <button id="copyBtn" onclick="copyToClipboard()" style="
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">📋 クリップボードにコピー（iPad推奨）</button>
        <span id="copyMsg" style="margin-left: 10px; color: #4CAF50; font-weight: bold;"></span>
    </div>

    <script>
    function copyToClipboard() {{
        const text = `{js_escaped_tsv}`;
        
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(showSuccess).catch(fallbackCopy);
        }} else {{
            fallbackCopy();
        }}

        function fallbackCopy() {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {{
                document.execCommand('copy');
                showSuccess();
            }} catch (err) {{
                alert('コピーに失敗しました。');
            }}
            document.body.removeChild(textArea);
        }}

        function showSuccess() {{
            const msg = document.getElementById("copyMsg");
            msg.innerText = "✓ コピーしました！";
            setTimeout(() => {{ msg.innerText = ""; }}, 3000);
        }}
    }}
    </script>
    """
    components.html(html_code, height=70)

    st.download_button(
        label="📥 CSVファイルとして保存",
        data=st.session_state["raw_csv_text"],
        file_name="blast_data.csv",
        mime="text/csv",
    )

    with st.expander("テキストを直接確認・編集する場合はこちら"):
        st.code(st.session_state["raw_tsv_text"], language="text")
