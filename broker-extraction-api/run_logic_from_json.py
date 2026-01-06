import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

from PIL import Image
from ultralytics import YOLO

from app.utils import classify_page_type
from app.services.position_processor import PositionProcessor
from app.services.transaction_processor import TransactionProcessor
from app.services.excel_exporter import excel_exporter


# ----------------------------
# Helpers
# ----------------------------
def load_paddle_json(json_path: str) -> Dict[int, List[dict]]:
    """
    Expected JSON format:
    [
      {"page": 1, "items": [ {"text": "...", "confidence": 0.99, "box": [[x,y],...]} , ... ]},
      {"page": 2, "items": [...]},
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(p["page"]): (p.get("items") or []) for p in data}


def items_to_ocr_result(items: List[dict]) -> List[dict]:
    """
    JSON item: {"text": str, "confidence": float, "box": [[x,y],...]}

    Convert -> format your processors expect:
    ocr_result = [{
        "rec_boxes": [(x1,y1,x2,y2), ...],
        "rec_texts": [...],
        "rec_scores": [...],
    }]
    """
    rec_boxes, rec_texts, rec_scores = [], [], []

    for it in items:
        text = (it.get("text", "") or "").strip()
        score = float(it.get("confidence", 0.0))
        pts = it.get("box") or []

        if pts and isinstance(pts, list) and len(pts) > 0:
            xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [p[1] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
            if xs and ys:
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                x1 = y1 = x2 = y2 = 0.0
        else:
            x1 = y1 = x2 = y2 = 0.0

        rec_boxes.append((float(x1), float(y1), float(x2), float(y2)))
        rec_texts.append(text)
        rec_scores.append(score)

    return [{
        "rec_boxes": rec_boxes,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores
    }]


def _natural_key(s: str) -> List[Any]:
    """
    Natural sort helper: page_2.png < page_10.png
    """
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_images_from_dir(images_dir: str) -> List[Path]:
    """
    Load image files from a directory, sorted naturally by filename.
    Supported: png, jpg, jpeg, webp, bmp, tif, tiff
    """
    p = Path(images_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No image files found in: {images_dir}")

    files_sorted = sorted(files, key=lambda x: _natural_key(x.name))
    return files_sorted


# ----------------------------
# Main pipeline (IMAGES + JSON)
# ----------------------------
def process_images_with_json(
    images_dir: str,
    json_path: str,
    out_dir: str | None = None,
    yolo_weights: str = "app/weights/yolo_broker_line_detect.pt",
) -> str:
    images_dir = os.path.abspath(images_dir)
    json_path = os.path.abspath(json_path)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    print(f"[CPU] IMAGES_DIR: {images_dir}")
    print(f"[CPU] JSON      : {json_path}")

    pages = load_paddle_json(json_path)

    print("[CPU] Loading image list...")
    image_paths = load_images_from_dir(images_dir)
    print(f"[CPU] Total images: {len(image_paths)}")

    print("[CPU] Loading YOLO weights...")
    yolo = YOLO(yolo_weights)

    pos_processor = PositionProcessor()
    tx_processor = TransactionProcessor()

    extracted = {"position": [], "transaction": {"trade": [], "fx_tf": [], "other": []}}

    for page_idx, img_path in enumerate(image_paths, start=1):
        items = pages.get(page_idx, [])
        if not items:
            print(f"[WARN] Page {page_idx}: no OCR items in JSON -> skip")
            continue

        # Open image
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"[WARN] Page {page_idx}: cannot open image {img_path.name} -> {e}")
            continue

        full_text = " ".join((i.get("text", "") or "") for i in items)
        page_type = classify_page_type(full_text)
        print(f"[CPU] Page {page_idx}/{len(image_paths)} ({img_path.name}) -> {page_type}")

        ocr_result = items_to_ocr_result(items)

        if page_type in ("position", "liquidity - accounts"):
            rows = pos_processor.process(yolo, image, ocr_result)
            extracted["position"].extend(rows)

        elif page_type == "transaction":
            out = tx_processor.process(yolo, image, ocr_result)
            extracted["transaction"]["trade"].extend(out.get("trade_info", []))
            extracted["transaction"]["fx_tf"].extend(out.get("fx_tf_info", []))
            extracted["transaction"]["other"].extend(out.get("other_info", []))

        else:
            # "other" -> ignore
            pass

    if out_dir is None:
        base = Path(json_path).stem.replace(".paddle", "")
        out_dir = os.path.join(os.getcwd(), "outputs_cpu", base)

    os.makedirs(out_dir, exist_ok=True)
    excel_exporter.export_to_excel(extracted, out_dir)

    print(f"[DONE] Excel files saved to: {out_dir}")
    return out_dir


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_logic_from_json.py <IMAGES_DIR> <JSON_PATH> [OUT_DIR]")
        raise SystemExit(1)

    images_dir_arg = sys.argv[1]
    json_path_arg = sys.argv[2]
    out_dir_arg = sys.argv[3] if len(sys.argv) >= 4 else None

    process_images_with_json(images_dir_arg, json_path_arg, out_dir_arg)
