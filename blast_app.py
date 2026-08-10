import cv2
import easyocr

# --------------------------------------------------
# 1. 指定された6項目の座標定義 (x1, y1, x2, y2)
# 左上座標(x1, y1) と 右下座標(x2, y2)
# ※「時刻」領域および余計な機能（CSV, グラフ, 異常検定）は削除済みです
# --------------------------------------------------
CROP_AREAS = {
    "バットスピード":     (0, 855, 355, 995),
    "アッパー度":       (355, 855, 710, 995),
    "オンプレーン効率":   (710, 855, 1067, 995),
    "加速度":           (0, 1185, 355, 1290),
    "スイング時間":     (355, 1165, 710, 1285),
    "パワー":           (710, 1165, 1067, 1285)
}

def read_batting_metrics(image_path: str) -> dict:
    """
    画像から指定された6項目のみを切り抜いて読み取る単純な関数
    """
    # OCRリーダーの初期化（数字・英字中心の場合は'en'で高速・高精度に設定）
    reader = easyocr.Reader(['en'], gpu=False)
    
    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像ファイルを読み込めませんでした: {image_path}")

    results = {}

    # 指定された6領域を切り出してOCR実行
    for item_name, (x1, y1, x2, y2) in CROP_AREAS.items():
        # 画像の切り出し (OpenCVは [y1:y2, x1:x2] の順番で指定)
        cropped = img[y1:y2, x1:x2]
        
        # OCR実行
        ocr_result = reader.readtext(cropped, detail=0)
        
        # 読み取り文字列の結合と整形
        if ocr_result:
            text = "".join(ocr_result).strip()
        else:
            text = None
            
        results[item_name] = text

    return results

# --------------------------------------------------
# 実行部
# --------------------------------------------------
if __name__ == "__main__":
    target_image = "sample_image.png"  # 対象の画像ファイルパス
    
    # 読み取り処理の実行
    metrics_data = read_batting_metrics(target_image)
    
    # 結果の表示（単純出力のみ）
    print("--- 読み取り結果 ---")
    for field_name, value in metrics_data.items():
        print(f"{field_name}: {value}")
