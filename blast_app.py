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

# 1. ページの設定
st.set_page_config(page_title="BLAST & Rapsodo データ統合", layout="wide")

st.title("BLAST & Rapsodo データ統合")
st.write(
    "Blast Motion のスクショ画像と、必要に応じて Rapsodo の CSV ファイルをアップロードしてください。"
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


def process_image_fast(uploaded_file, index):
    """画像解析処理（並列処理用にインデックスを保持）"""
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

        for metric, (min_val, max_val) in METRIC_RANGES.items():
            val = row.get(metric)
            if pd.notna(val) and isinstance(val, (int, float)):
                if val < min_val or val > max_val:
                    reasons.append(f"{metric}範囲外({val})")

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

    if "判定" in df.columns:
        df["判定"] = status_list
    else:
        df.insert(1, "判定", status_list)
    return df


def merge_rapsodo_data(blast_df, rapsodo_file, anchor_blast_index, max_time_diff_sec=30):
    """アンカー（基準スイング）を指定して時刻補正後、Rapsodoデータを結合"""
    try:
        rap_df = pd.read_csv(rapsodo_file)
    except Exception as e:
        st.error(f"Rapsodo CSVの読み込みに失敗しました: {e}")
        return blast_df

    # 列名の正規化（大文字小文字/スペースの揺れを修正）
    rap_cols = {col.strip().lower(): col for col in rap_df.columns}
    
    date_col = rap_cols.get("date")
    ev_col = rap_cols.get("exitvelocity") or rap_cols.get("exit velocity") or rap_cols.get("exit_velocity")
    la_col = rap_cols.get("launchangle") or rap_cols.get("launch angle") or rap_cols.get("launch_angle")
    sd_col = rap_cols.get("spin direct") or rap_cols.get("spin direction") or rap_cols.get("spin_direction") or rap_cols.get("spindirection")

    if not date_col:
        st.warning("⚠️ RapsodoのCSV内に『Date』列が見つかりませんでした。")
        return blast_df

    # Rapsodoの時刻変換
    rap_df["_dt_orig"] = pd.to_datetime(rap_df[date_col], errors="coerce")
    if rap_df["_dt_orig"].isna().all():
        rap_df["_dt_orig"] = pd.to_datetime("2020-01-01 " + rap_df[date_col].astype(str), errors="coerce")

    # Blastの時刻変換
    blast_df["_dt"] = pd.to_datetime("2020-01-01 " + blast_df["時刻"].astype(str), errors="coerce")

    valid_rap = rap_df.dropna(subset=["_dt_orig"]).sort_values("_dt_orig").reset_index(drop=True)
    valid_blast = blast_df.dropna(subset=["_dt"])

    if valid_rap.empty or valid_blast.empty:
        st.warning("⚠️ BlastまたはRapsodoの時刻データを正しく読み込めませんでした。")
        blast_df.drop(columns=["_dt"], inplace=True, errors="ignore")
        return blast_df

    # 指定された Blast スイング（1始まり -> 0インデックス）の時刻取得
    anchor_idx = anchor_blast_index - 1
    if anchor_idx < 0 or anchor_idx >= len(blast_df):
        anchor_idx = 0

    target_blast_dt = blast_df.iloc[anchor_idx]["_dt"]
    first_rap_dt = valid_rap.iloc[0]["_dt_orig"]

    if pd.isna(target_blast_dt) or pd.isna(first_rap_dt):
        time_offset = pd.Timedelta(0)
    else:
        # オフセット（時間のズレ量）の計算: Blast指定時刻 - Rapsodo1球目時刻
        time_offset = target_blast_dt - first_rap_dt

    # Rapsodo側の時刻を一括補正
    valid_rap["_dt_corrected"] = valid_rap["_dt_orig"] + time_offset

    # 結合用初期カラムのセット
    blast_df["Exit Velocity"] = np.nan
    blast_df["Launch Angle"] = np.nan
    blast_df["Spin Direction"] = np.nan

    # 許容時間差（デフォルト30秒）以内で最も近いデータをマッチング
    for idx, row in blast_df.iterrows():
        b_dt = row["_dt"]
        if pd.notna(b_dt):
            diffs = (valid_rap["_dt_corrected"] - b_dt).abs()
            min_diff_idx = diffs.idxmin()
            min_diff = diffs.loc[min_diff_idx]

            if min_diff.total_seconds() <= max_time_diff_sec:
                closest_row = valid_rap.loc[min_diff_idx]
                if ev_col and ev_col in closest_row:
                    blast_df.at[idx, "Exit Velocity"] = closest_row[ev_col]
                if la_col and la_col in closest_row:
                    blast_df.at[idx, "Launch Angle"] = closest_row[la_col]
                if sd_col and sd_col in closest_row:
                    blast_df.at[idx, "Spin Direction"] = closest_row[sd_col]

    offset_seconds = int(time_offset.total_seconds())
    offset_min = abs(offset_seconds) // 60
    offset_sec = abs(offset_seconds) % 60
    direction = "進み" if offset_seconds < 0 else "遅れ"
    
    st.success(
        f"⏱️ 補正完了: Rapsodoの時計が Blast より **{offset_min}分{offset_sec}秒 {direction}** いていたと判定して結合しました。"
    )

    blast_df.drop(columns=["_dt"], inplace=True, errors="ignore")
    return blast_df


def render_summary_metrics(df):
    """平均値および最高値の集計枠を表示"""
    st.markdown("#### 📊 スイングデータ要約（平均・最高）")

    metrics_keys = ["バットスピード", "アッパー度", "オンプレーン効率", "加速度", "スイング時間", "パワー"]
    if "Exit Velocity" in df.columns:
        metrics_keys.extend(["Exit Velocity", "Launch Angle", "Spin Direction"])
    
    avg_row = {"区分": "平均値"}
    best_row = {"区分": "最高値"}

    for key in metrics_keys:
        if key in df.columns:
            series = pd.to_numeric(df[key], errors="coerce").dropna()
            
            if not series.empty:
                if key in ["スイング時間"]:
                    avg_row[key] = f"{series.mean():.2f}"
                else:
                    avg_row[key] = f"{series.mean():.1f}"

                if key in ["バットスピード", "オンプレーン効率", "加速度", "パワー", "Exit Velocity", "Launch Angle"]:
                    best_row[key] = f"{series.max():.1f}"
                elif key == "スイング時間":
                    best_row[key] = f"{series.min():.2f}"
                elif key == "アッパー度":
                    avg_row[key] = f"{series.mean():.1f}"
                    best_row[key] = f"{series.max():.1f} / {series.min():.1f}"
                else:
                    best_row[key] = "-"
            else:
                avg_row[key] = "-"
                best_row[key] = "-"
        else:
            avg_row[key] = "-"
            best_row[key] = "-"

    summary_df = pd.DataFrame([avg_row, best_row]).set_index("区分")
    st.table(summary_df)


def render_time_series_chart(df):
    """時系列推移グラフ"""
    st.markdown("### 📈 時系列データの推移")

    plot_df = df.copy()
    if "時刻" in plot_df.columns and plot_df["時刻"].notna().any():
        plot_df["X軸ラベル"] = plot_df.apply(
            lambda r: f"{r.name + 1}回目 ({r['時刻']})" if pd.notna(r["時刻"]) else f"{r.name + 1}回目",
            axis=1
        )
    else:
        plot_df["X軸ラベル"] = [f"{i + 1}回目" for i in range(len(plot_df))]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("【主要指標】スピード・効率・パワー・Exit Velocity", "【詳細指標】アッパー度・加速度・スイング時間・Launch Angle"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    colors = {
        "バットスピード": "#1f77b4",
        "オンプレーン効率": "#2ca02c",
        "アッパー度": "#ff7f0e",
        "加速度": "#9467bd",
        "パワー": "#d62728",
        "スイング時間": "#17becf",
        "Exit Velocity": "#e377c2",
        "Launch Angle": "#8c564b"
    }

    if "バットスピード" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["バットスピード"], errors="coerce"),
                name="バットスピード(km/h)",
                mode="lines+markers",
                line=dict(width=2, color=colors["バットスピード"]),
                marker=dict(size=6),
            ),
            row=1, col=1, secondary_y=False
        )

    if "オンプレーン効率" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["オンプレーン効率"], errors="coerce"),
                name="オンプレーン効率(%)",
                mode="lines+markers",
                line=dict(width=2, color=colors["オンプレーン効率"]),
                marker=dict(size=6),
            ),
            row=1, col=1, secondary_y=False
        )

    if "Exit Velocity" in plot_df.columns and plot_df["Exit Velocity"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["Exit Velocity"], errors="coerce"),
                name="Exit Velocity",
                mode="lines+markers",
                line=dict(width=2, color=colors["Exit Velocity"]),
                marker=dict(size=6),
            ),
            row=1, col=1, secondary_y=False
        )

    if "パワー" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["パワー"], errors="coerce"),
                name="パワー(Kw) [右軸]",
                mode="lines+markers",
                line=dict(width=2, dash="dot", color=colors["パワー"]),
                marker=dict(size=6),
            ),
            row=1, col=1, secondary_y=True
        )

    if "アッパー度" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["アッパー度"], errors="coerce"),
                name="アッパー度(deg)",
                mode="lines+markers",
                line=dict(width=2.5, color=colors["アッパー度"]),
                marker=dict(size=7),
            ),
            row=2, col=1, secondary_y=False
        )

    if "加速度" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["加速度"], errors="coerce"),
                name="加速度(G)",
                mode="lines+markers",
                line=dict(width=2.5, color=colors["加速度"]),
                marker=dict(size=7),
            ),
            row=2, col=1, secondary_y=False
        )

    if "Launch Angle" in plot_df.columns and plot_df["Launch Angle"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["Launch Angle"], errors="coerce"),
                name="Launch Angle(deg)",
                mode="lines+markers",
                line=dict(width=2.5, color=colors["Launch Angle"]),
                marker=dict(size=7),
            ),
            row=2, col=1, secondary_y=False
        )

    if "スイング時間" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["X軸ラベル"],
                y=pd.to_numeric(plot_df["スイング時間"], errors="coerce"),
                name="スイング時間(sec) [右軸]",
                mode="lines+markers",
                line=dict(width=2.5, dash="dash", color=colors["スイング時間"]),
                marker=dict(size=7),
            ),
            row=2, col=1, secondary_y=True
        )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=650,
    )

    fig.update_yaxes(title_text="スピード / 効率 / EV", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="パワー", row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="角度(deg) / 加速度(G)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="スイング時間(sec)", row=2, col=1, secondary_y=True)

    fig.update_xaxes(title_text="スイング順 (時刻)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# --- 画面UI部分 ---
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "1. BLASTのスクショ画像を選択（複数選択可）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

with col2:
    rapsodo_file = st.file_uploader(
        "2. Rapsodoデータ (CSV) を追加（任意）",
        type=["csv"],
        accept_multiple_files=False,
    )
    
    # Rapsodo CSV が選択された時だけ基準スイング番号の入力欄を表示
    anchor_blast_index = 1
    if rapsodo_file is not None:
        anchor_blast_index = st.number_input(
            "🎯 Rapsodoの1球目は、BLASTの何スイング目にあたりますか？",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="例：空振り等でRapsodoの1計測目がBLASTの2スイング目の場合、「2」と入力してください。"
        )

if uploaded_files:
    if st.button("読み取る"):
        raw_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(uploaded_files)
        completed_count = 0

        # 並列処理（マルチスレッド）の実行
        max_workers = min(4, total)
        status_text.text(f"🚀 並列処理中 (最大{max_workers}画像を同時解析)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_image_fast, file, idx): file
                for idx, file in enumerate(uploaded_files)
            }

            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    data = future.result()
                    raw_results.append(data)
                except Exception as e:
                    st.error(f"ファイル '{file.name}' の処理中にエラーが発生しました: {e}")

                completed_count += 1
                progress_bar.progress(completed_count / total)
                status_text.text(f"解析完了: {completed_count}/{total} 枚")

        if raw_results:
            status_text.text("✅ すべての画像の解析が完了しました！")
            raw_results.sort(key=lambda x: x["_index"])
            for r in raw_results:
                del r["_index"]

            df = pd.DataFrame(raw_results)
            df = check_outliers(df)

            # Rapsodo CSV が追加されている場合はアンカー基準で時間統合を実施
            if rapsodo_file is not None:
                df = merge_rapsodo_data(df, rapsodo_file, anchor_blast_index)

            st.session_state["parsed_df"] = df

if "parsed_df" in st.session_state:
    current_df = st.session_state["parsed_df"]

    has_anomaly = current_df["判定"].str.contains("⚠️").any()

    def render_editor():
        st.subheader("【解析結果一覧・手修正】")
        if has_anomaly:
            st.warning("⚠️ 異常と思われる値が検出されています。表のセルを直接修正・削除してください。")
        else:
            st.info("💡 画面上のセルを直接ダブルタップ/ダブルクリックして数値を変更・修正できます。")

        render_summary_metrics(current_df)

        edited_df = st.data_editor(
            current_df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )
        
        st.session_state["parsed_df"] = edited_df

        render_time_series_chart(edited_df)

        return edited_df

    def render_excel_copy(df_to_export):
        st.markdown("### 📋 Excelへ一括コピー")
        st.write(
            "下の枠内のデータ（数値データ＋時刻／ヘッダーなし）を全選択してコピー（Ctrl+C）し、Excelのセルにそのまま貼り付けてください（Ctrl+V）。"
        )

        export_df = df_to_export.drop(columns=["ファイル名", "判定"], errors="ignore")
        tsv_data = export_df.to_csv(index=False, header=False, sep="\t")
        
        st.code(tsv_data, language="text")

        csv_data = df_to_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 CSVファイルとして保存",
            data=csv_data,
            file_name="blast_rapsodo_combined_data.csv",
            mime="text/csv",
        )

    if has_anomaly:
        latest_df = render_editor()
        render_excel_copy(latest_df)
    else:
        render_excel_copy(current_df)
        render_editor()
