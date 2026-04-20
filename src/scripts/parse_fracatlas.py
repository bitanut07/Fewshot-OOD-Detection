from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
ANNOTATION_FILE = "data/raw/FracAtlas/dataset.csv"
IMAGE_ROOT = Path("data/raw/FracAtlas/images")
OUTPUT_CSV = "data/processed/data_processing/fracatlas.csv"
DATASET_NAME = "FracAtlas"

IMAGE_COL = "image_id"
LABEL_COL = None
FRACTURE_FLAG_COL = "fractured"

POSITIVE_VALUES = {"1", "true", "yes", "y", "fractured", "positive"}

POSITIVE_LABEL = "Fractured"
NEGATIVE_LABEL = "Non-fractured"

USED_LABELS = [
    "fractured",
]

UNUSED_LABELS = [
    "non-fractured",
]

DEFAULT_NO_LABEL = "no_tumor"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x) -> str:
    return str(x).strip()


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


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


def infer_raw_label(row: pd.Series) -> str:
    if LABEL_COL and LABEL_COL in row.index:
        return normalize_text(row[LABEL_COL])

    if FRACTURE_FLAG_COL not in row.index:
        raise ValueError(
            f"Neither LABEL_COL nor FRACTURE_FLAG_COL found. "
            f"Available columns: {list(row.index)}"
        )

    value = normalize_text(row[FRACTURE_FLAG_COL]).lower()
    if value in POSITIVE_VALUES:
        return POSITIVE_LABEL
    return NEGATIVE_LABEL


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ann_path = Path(ANNOTATION_FILE)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root not found: {IMAGE_ROOT}")

    df = pd.read_csv(ann_path)

    if IMAGE_COL not in df.columns:
        raise ValueError(
            f"Column '{IMAGE_COL}' not found in {ANNOTATION_FILE}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []
    skipped_missing_image = 0

    for _, row in df.iterrows():
        image_name_raw = normalize_text(row[IMAGE_COL])
        if not image_name_raw or image_name_raw.lower() == "nan":
            continue

        raw_label = infer_raw_label(row)
        norm_label = normalize_label(raw_label)

        img_path = find_image_path(IMAGE_ROOT, image_name_raw)
        if img_path is None:
            skipped_missing_image += 1
            print(f"[WARN] missing image: {image_name_raw}")
            continue

        image_stem = Path(image_name_raw).stem

        if norm_label in USED_LABELS:
            final_label = norm_label
        elif norm_label in UNUSED_LABELS:
            continue
        else:
            final_label = DEFAULT_NO_LABEL

        rows.append(
            {
                "image_name": image_stem,
                "path": str(img_path),
                "data": DATASET_NAME,
                "label": final_label,
            }
        )

    out_df = pd.DataFrame(rows, columns=["image_name", "path", "data", "label"])
    ensure_dir(Path(OUTPUT_CSV).parent)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"[OK] wrote: {OUTPUT_CSV}")
    print(f"[INFO] rows: {len(out_df)}")
    print(f"[INFO] skipped_missing_image: {skipped_missing_image}")

    if not out_df.empty:
        print("\n[INFO] label counts:")
        print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()