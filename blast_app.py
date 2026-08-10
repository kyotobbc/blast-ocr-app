import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. ページ設定
st.set_page_config(page_title="BLAST & Rapsodo データ統合", layout="wide")
st.title("BLAST & Rapsodo データ統合")
st.write("Blast Motion のスクショ画像と、必要に応じて Rapsodo の CSV ファイルをアップロードしてください。")

# 2. OCR初期化
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False, quantize=True)

reader = load_ocr()

# 3. 定数定義
NUMERIC_AREAS = {
    "バットスピード": (0, 855, 355, 995),
    "アッパー度": (355, 855, 710, 995),
    "オンプレーン効率": (710, 855, 1067, 995),
    "加速度": (0, 1185, 355, 1290),
    "スイング時間": (355, 1165, 710, 1285),
    "パワー": (710, 1165, 1067, 1285),
    "時刻": (525, 45, 960, 115),
}

METRIC_RANGES = {
    "バットスピード": (30.0, 170.0),
    "アッパー度": (-30.0, 45.0),
    "オンプレーン効率": (20.0, 100.0),
    "加速度": (3.0, 35.0),
    "スイング時間": (0.08, 0.45),
    "パワー": (0.5, 12.0),
}


def fix_numeric_format(val, metric_name):
    """OCRテキストの整形とフォーマット変換"""
    if val is None:
        return None
    val_str = str(val).strip()

    if metric_name == "時刻":
        cleaned = re.sub(r"[.,;]", ":", val_str.translate(str.maketrans("OoIil", "00111")))
        match = re.search(r"(\d{1,2}:\d{2}:\d{2})", cleaned)
        return match.group(1) if match else None

    digits_only = re.sub(r"\D", "", val_str)
    try:
        if metric_name == "加速度":
            if len(digits_only) == 3: return float(f"{digits_only[:2]}.{digits_only[2:]}")
            if len(digits_only) == 2: return float(f"{digits_only}.0")
            return round(float(val_str), 1)
        elif metric_name == "スイング時間":
            if len(digits_only) >= 2: return float(f"0.{digits_only[-2:]}")
            return round(float(val_str), 2)
        return float(val_str)
    except ValueError:
        return val


def process_image_fast(uploaded_file, index):
    """画像解析処理"""
    img = Image.open(uploaded_file).convert("RGB")
    open_cv_full = np.array(img, dtype=np.uint8)
    results = {"_index": index, "ファイル名": uploaded_file.name}

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
            raw_ocr = "".join(reader.readtext(thresh, detail=0, allowlist="0123456789:.,oOIilS "))
            results[metric_name] = fix_numeric_format(raw_ocr, metric_name)
        else:
            large = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)
            raw_ocr = "".join(reader.readtext(gray, detail=0, allowlist="0123456789.-", mag_ratio=1.0))
            match = re.search(r"-?\d+\.?\d*", raw_ocr)
            results[metric_name] = fix_numeric_format(match.group(0), metric_name) if match else None

    return results


def check_outliers(df):
    """判定フラグ設定"""
    status_list = []
    for _, row in df.iterrows():
        reasons = []
        for metric, (min_val, max_val) in METRIC_RANGES.items():
            val = row.get(metric)
            if pd.notna(val) and isinstance(val, (int, float)):
                if val < min_val or val > max_val:
                    reasons.append(f"{metric}範囲外({val})")
                col_data = pd.to_numeric(df[metric], errors="coerce").dropna()
                if len(col_data) >= 4:
                    std = col_data.std()
                    if std > 0 and abs(val - col_data.mean()) > 3 * std:
                        reasons.append(f"{metric}外れ値({val})")
        status_list.append("⚠️ " + ", ".join(reasons) if reasons else "✅ 正常")

    df.insert(1, "判定", status_list)
    return df


def merge_rapsodo_data(blast_df, rapsodo_file, anchor_blast_index, max_time_diff_sec=30):
    """日付を無視し時間のみで時刻補正・Rapsodoデータの結合"""
    try:
        rap_df = pd.read_csv(rapsodo_file)
    except Exception as e:
        st.error(f"Rapsodo CSVの読み込みに失敗しました: {e}")
        return blast_df

    rap_cols = {col.strip().lower().replace(" ", "_"): col for col in rap_df.columns}
    date_col = rap_cols.get("date")
    ev_col = rap_cols.get("exitvelocity") or rap_cols.get("exit_velocity")
    la_col = rap_cols.get("launchangle") or rap_cols.get("launch_angle")
    sd_col = rap_cols.get("spin_direct") or rap_cols.get("spin_direction") or rap_cols.get("spindirection")

    if not date_col:
        st.warning("⚠️ RapsodoのCSV内に『Date』列が見つかりませんでした。")
        return blast_df

    # RapsodoのDate列から「時間（HH:MM:SS）」だけを抽出して統一基準日(2020-01-01)を付与
    def extract_time_str(val):
        if pd.isna(val):
            return None
        val_s = str(val).strip()
        # HH:MM:SS または HH:MM:SS.fff のパターンを抽出
        match = re.search(r"(\d{1,2}:\d{2}:\d{2})", val_s)
        return match.group(1) if match else None

    rap_time_strs = rap_df[date_col].apply(extract_time_str)
    rap_df["_dt_orig"] = pd.to_datetime("2020-01-01 " + rap_time_strs, errors="coerce")

    # Blast側も同様に2020-01-01基準のDatetime化
    blast_df["_dt"] = pd.to_datetime("2020-01-01 " + blast_df["時刻"].astype(str), errors="coerce")

    valid_rap = rap_df.dropna(subset=["_dt_orig"]).sort_values("_dt_orig").reset_index(drop=True)
    if valid_rap.empty or blast_df["_dt"].dropna().empty:
        st.warning("⚠️ 時刻データを正しく読み込めませんでした。")
        return blast_df.drop(columns=["_dt"], errors="ignore")

    anchor_idx = max(0, min(anchor_blast_index - 1, len(blast_df) - 1))
    target_blast_dt = blast_df.iloc[anchor_idx]["_dt"]
    first_rap_dt = valid_rap.iloc[0]["_dt_orig"]

    time_offset = (target_blast_dt - first_rap_dt) if pd.notna(target_blast_dt) and pd.notna(first_rap_dt) else pd.Timedelta(0)
    valid_rap["_dt_corrected"] = valid_rap["_dt_orig"] + time_offset

    for col in ["Exit Velocity", "Launch Angle", "Spin Direction"]:
        blast_df[col] = np.nan

    for idx, row in blast_df.iterrows():
        b_dt = row["_dt"]
        if pd.notna(b_dt):
            diffs = (valid_rap["_dt_corrected"] - b_dt).abs()
            min_idx = diffs.idxmin()
            if diffs.loc[min_idx].total_seconds() <= max_time_diff_sec:
                closest = valid_rap.loc[min_idx]
                if ev_col: blast_df.at[idx, "Exit Velocity"] = closest.get(ev_col)
                if la_col: blast_df.at[idx, "Launch Angle"] = closest.get(la_col)
                if sd_col: blast_df.at[idx, "Spin Direction"] = closest.get(sd_col)

    offset_sec = int(time_offset.total_seconds())
    st.success(f"⏱️ 補正完了: Rapsodoの時計が Blast より **{abs(offset_sec)//60}分{abs(offset_sec)%60}秒 {'進み' if offset_sec < 0 else '遅れ'}** いていたと判定して結合しました。")
    return blast_df.drop(columns=["_dt"], errors="ignore")


def render_summary_metrics(df):
    """サマリー表示"""
    st.markdown("#### 📊 スイングデータ要約（平均・最高）")
    metrics_keys = ["バットスピード", "アッパー度", "オンプレーン効率", "加速度", "スイング時間", "パワー", "Exit Velocity", "Launch Angle", "Spin Direction"]

    avg_row, best_row = {"区分": "平均値"}, {"区分": "最高値"}
    for key in metrics_keys:
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
            if not s.empty:
                avg_row[key] = f"{s.mean():.2f}" if key == "スイング時間" else f"{s.mean():.1f}"
                if key in ["バットスピード", "オンプレーン効率", "加速度", "パワー", "Exit Velocity", "Launch Angle"]:
                    best_row[key] = f"{s.max():.1f}"
                elif key == "スイング時間":
                    best_row[key] = f"{s.min():.2f}"
                elif key == "アッパー度":
                    best_row[key] = f"{s.max():.1f} / {s.min():.1f}"
                else:
                    best_row[key] = "-"
            else:
                avg_row[key] = best_row[key] = "-"
        else:
            avg_row[key] = best_row[key] = "-"

    st.table(pd.DataFrame([avg_row, best_row]).set_index("区分"))


def render_time_series_chart(df):
    """時系列グラフ描画"""
    st.markdown("### 📈 時系列データの推移")
    plot_df = df.copy()
    plot_df["X軸ラベル"] = [
        f"{i+1}回目 ({t})" if pd.notna(t) else f"{i+1}回目" 
        for i, t in enumerate(plot_df.get("時刻", [None]*len(plot_df)))
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=("【主要指標】スピード・効率・パワー・Exit Velocity", "【詳細指標】アッパー度・加速度・スイング時間・Launch Angle"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    traces_config = [
        ("バットスピード", "(km/h)", 1, 1, False, None, "#1f77b4"),
        ("オンプレーン効率", "(%)", 1, 1, False, None, "#2ca02c"),
        ("Exit Velocity", "", 1, 1, False, None, "#e377c2"),
        ("パワー", "(Kw)", 1, 1, True, "dot", "#d62728"),
        ("アッパー度", "(deg)", 2, 1, False, None, "#ff7f0e"),
        ("加速度", "(G)", 2, 1, False, None, "#9467bd"),
        ("Launch Angle", "(deg)", 2, 1, False, None, "#8c564b"),
        ("スイング時間", "(sec)", 2, 1, True, "dash", "#17becf"),
    ]

    for name, unit, r, c, sec_y, dash, color in traces_config:
        if name in plot_df.columns and plot_df[name].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df["X軸ラベル"], y=pd.to_numeric(plot_df[name], errors="coerce"),
                    name=f"{name}{unit}" + (" [右軸]" if sec_y else ""),
                    mode="lines+markers", line=dict(width=2, dash=dash, color=color), marker=dict(size=6)
                ),
                row=r, col=c, secondary_y=sec_y
            )

    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1), margin=dict(l=20, r=20, t=60, b=20), height=650)
    fig.update_yaxes(title_text="スピード / 効率 / EV", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="パワー", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="角度(deg) / 加速度(G)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="スイング時間(sec)", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="スイング順 (時刻)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# --- 画面UI ---
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("1. BLASTのスクショ画像を選択（複数選択可）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
with col2:
    rapsodo_file = st.file_uploader("2. Rapsodoデータ (CSV) を追加（任意）", type=["csv"], accept_multiple_files=False)
    anchor_blast_index = 1
    if rapsodo_file is not None:
        anchor_blast_index = st.number_input(
            "🎯 Rapsodoの1球目は、BLASTの何スイング目にあたりますか？",
            min_value=1, max_value=100, value=1, step=1,
            help="例：空振り等でRapsodoの1計測目がBLASTの2スイング目の場合、「2」と入力してください。"
        )

if uploaded_files and st.button("読み取る"):
    raw_results, total = [], len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    max_workers = min(4, total)

    status_text.text(f"🚀 並列処理中 (最大{max_workers}画像を同時解析)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_fast, file, idx): file for idx, file in enumerate(uploaded_files)}
        for completed_count, future in enumerate(as_completed(futures), 1):
            try:
                raw_results.append(future.result())
            except Exception as e:
                st.error(f"エラー: {futures[future].name} - {e}")
            progress_bar.progress(completed_count / total)
            status_text.text(f"解析完了: {completed_count}/{total} 枚")

    if raw_results:
        status_text.text("✅ 解析完了")
        raw_results.sort(key=lambda x: x["_index"])
        for r in raw_results: del r["_index"]

        df = check_outliers(pd.DataFrame(raw_results))
        if rapsodo_file is not None:
            df = merge_rapsodo_data(df, rapsodo_file, anchor_blast_index)

        st.session_state["parsed_df"] = df

if "parsed_df" in st.session_state:
    current_df = st.session_state["parsed_df"]
    has_anomaly = current_df["判定"].str.contains("⚠️").any()

    def render_editor():
        st.subheader("【解析結果一覧・手修正】")
        st.warning("⚠️ 異常と思われる値が検出されています。表のセルを直接修正・削除してください。") if has_anomaly else st.info("💡 セルを直接クリックして手修正可能です。")
        render_summary_metrics(current_df)
        edited_df = st.data_editor(current_df, num_rows="dynamic", use_container_width=True, key="data_editor")
        st.session_state["parsed_df"] = edited_df
        render_time_series_chart(edited_df)
        return edited_df

    def render_excel_copy(df_to_export):
        st.markdown("### 📋 Excelへ一括コピー")
        st.write("下の枠内のデータを全選択してコピー（Ctrl+C）し、Excelのセルに貼り付けてください（Ctrl+V）。")
        export_df = df_to_export.drop(columns=["ファイル名", "判定"], errors="ignore")
        st.code(export_df.to_csv(index=False, header=False, sep="\t"), language="text")
        st.download_button("📥 CSVファイルとして保存", df_to_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "blast_rapsodo_combined.csv", "text/csv")

    if has_anomaly:
        latest_df = render_editor()
        render_excel_copy(latest_df)
    else:
        render_excel_copy(current_df)
        render_editor()
