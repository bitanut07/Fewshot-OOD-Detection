from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
ANNOTATION_FILE = "data/raw/BTXRD/dataset.xlsx"
IMAGE_ROOT = Path("data/raw/BTXRD/images")
OUTPUT_CSV = "data/processed/data_processing/btxrd.csv"
DATASET_NAME = "BTXRD"

IMAGE_COL = "image_id"

# Các label muốn giữ
USED_LABELS = [
    "giant cell tumor",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "simple bone cyst",
    "synovial osteochondroma",
]

# Các label không sử dụng
UNUSED_LABELS = [
    "multiple osteochondromas",
    "other bt",
    "other mt",
]

# Nếu ảnh không thuộc USED_LABELS và cũng không thuộc UNUSED_LABELS
# thì gán label này
DEFAULT_NO_LABEL = "no_tumor"

# Nếu True:
# - ảnh có nhiều hơn 1 label active trong USED_LABELS sẽ bị bỏ
# Nếu False:
# - mỗi label active trong USED_LABELS sẽ tạo 1 dòng riêng
DROP_MULTI_LABEL = False

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x) -> str:
    return str(x).strip()


def is_active_label_value(value) -> bool:
    if pd.isna(value):
        return False

    try:
        return int(value) == 1
    except Exception:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_image_path(image_root: Path, image_name: str) -> Path | None:
    direct_path = image_root / image_name
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    stem = Path(image_name).stem

    for ext in IMAGE_EXTS:
        candidate = image_root / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    for ext in IMAGE_EXTS:
        matches = list(image_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def get_active_labels(row: pd.Series, labels: list[str]) -> list[str]:
    active = []
    for label in labels:
        if label not in row.index:
            continue
        if is_active_label_value(row[label]):
            active.append(label)
    return active


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ann_path = Path(ANNOTATION_FILE)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root not found: {IMAGE_ROOT}")

    df = pd.read_excel(ann_path)

    if IMAGE_COL not in df.columns:
        raise ValueError(
            f"Column '{IMAGE_COL}' not found in {ANNOTATION_FILE}. "
            f"Available columns: {list(df.columns)}"
        )

    missing_used = [x for x in USED_LABELS if x not in df.columns]
    missing_unused = [x for x in UNUSED_LABELS if x not in df.columns]

    if missing_used:
        print(f"[WARN] USED_LABELS not found in xlsx columns: {missing_used}")
    if missing_unused:
        print(f"[WARN] UNUSED_LABELS not found in xlsx columns: {missing_unused}")

    rows = []
    skipped_multi_label = 0
    skipped_missing_image = 0

    for _, row in df.iterrows():
        image_name_raw = normalize_text(row[IMAGE_COL])
        if not image_name_raw or image_name_raw.lower() == "nan":
            continue

        active_used = get_active_labels(row, USED_LABELS)
        active_unused = get_active_labels(row, UNUSED_LABELS)

        img_path = find_image_path(IMAGE_ROOT, image_name_raw)
        if img_path is None:
            skipped_missing_image += 1
            print(f"[WARN] missing image: {image_name_raw}")
            continue

        image_stem = Path(image_name_raw).stem

        # Case 1: có label trong USED_LABELS
        if len(active_used) > 0:
            if DROP_MULTI_LABEL and len(active_used) != 1:
                skipped_multi_label += 1
                continue

            for label in active_used:
                rows.append(
                    {
                        "image_name": image_stem,
                        "path": str(img_path),
                        "data": DATASET_NAME,
                        "label": label,
                    }
                )
            continue

        # Case 2: không có USED, có UNUSED -> bỏ
        if len(active_unused) > 0:
            continue

        # Case 3: không có cả USED lẫn UNUSED -> gán no_tumor
        rows.append(
            {
                "image_name": image_stem,
                "path": str(img_path),
                "data": DATASET_NAME,
                "label": DEFAULT_NO_LABEL,
            }
        )

    out_df = pd.DataFrame(rows, columns=["image_name", "path", "data", "label"])
    ensure_dir(Path(OUTPUT_CSV).parent)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"[OK] wrote: {OUTPUT_CSV}")
    print(f"[INFO] total rows: {len(out_df)}")
    print(f"[INFO] skipped_multi_label: {skipped_multi_label}")
    print(f"[INFO] skipped_missing_image: {skipped_missing_image}")

    if not out_df.empty:
        print("\n[INFO] kept label counts:")
        print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()