from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

# Input CSVs
BTXRD_CSV = "data/processed/data_processing/btxrd.csv"
FRACATLAS_CSV = "data/processed/data_processing/fracatlas.csv"

# Output
DATASET_NAME = "splits_dataset"
PROCESSED_ROOT = Path("data/processed/image_processing")
IMAGES_DIR = PROCESSED_ROOT 
MANIFEST_CSV = "data/processed/data_processing/manifest.csv"

# Folder skip images
SKIPPED_ROOT = Path("data/processed/skip_images")
SKIPPED_DIR = SKIPPED_ROOT
SKIPPED_CSV = SKIPPED_ROOT / "skipped.csv"

# Label settings
ID_LABELS = [
    "giant cell tumor",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "simple bone cyst",
    "synovial osteochondroma",
]

OOD_LABELS = [
    "no_tumor",
    "fractured",
]

# Source code mapping for name_id
SOURCE_CODE_MAP = {
    "BTXRD": "01",
    "FracAtlas": "02",
}

# Image filtering / preprocessing
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_HEIGHT = 128
MIN_WIDTH = 128

# White-ratio filter
WHITE_THRESH = 245
WHITE_RATIO_THRESHOLD = 0.3


# Diversity code
DIVERSITY_CODES = {
    "orig": "00",
    "rot90": "01",
    "rot180": "02",
    "rot270": "03",
}

# Final save size
SAVE_SIZE = (224, 224)


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x) -> str:
    return str(x).strip()


def load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)

    required_cols = {"image_name", "path", "data", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["image_name"] = df["image_name"].map(normalize_text)
    df["path"] = df["path"].map(normalize_text)
    df["data"] = df["data"].map(normalize_text)
    df["label"] = df["label"].map(normalize_text).str.lower()

    return df


def assign_class(label: str) -> str | None:
    if label in ID_LABELS:
        return "id"
    if label in OOD_LABELS:
        return "ood"
    return None


def build_base_name_id(source_dataset: str, index: int) -> str:
    if source_dataset not in SOURCE_CODE_MAP:
        raise ValueError(f"Unknown source dataset: {source_dataset}")
    source_code = SOURCE_CODE_MAP[source_dataset]
    return f"IMG{source_code}{index:05d}"


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def resize_keep_output(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def compute_white_ratio(img: np.ndarray, white_thresh: int = WHITE_THRESH) -> float:
    white_pixels = np.sum(img >= white_thresh)
    total_pixels = img.shape[0] * img.shape[1]
    return float(white_pixels) / float(total_pixels)


def passes_xray_filter(
    img: np.ndarray,
    white_thresh: int = WHITE_THRESH,
    white_ratio_threshold: float = WHITE_RATIO_THRESHOLD,
) -> bool:
    if img is None:
        return False

    h, w = img.shape[:2]
    if h < MIN_HEIGHT or w < MIN_WIDTH:
        return False

    white_ratio = compute_white_ratio(img, white_thresh=white_thresh)
    return white_ratio <= white_ratio_threshold


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def generate_variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants = [
        ("orig", img),
        ("rot90", rotate_image(img, 90)),
        ("rot180", rotate_image(img, 180)),
        ("rot270", rotate_image(img, 270)),
    ]

    return variants


def save_image(img: np.ndarray, out_path: Path) -> None:
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {out_path}")
    

# Chuyển ảnh vào folder skip
def copy_to_skip_folder(src_path: Path, skip_dir: Path) -> str:
    ensure_dir(skip_dir)

    if not src_path.exists() or not src_path.is_file():
        return ""

    dst_path = skip_dir / src_path.name
    stem = src_path.stem
    suffix = src_path.suffix
    counter = 1

    while dst_path.exists():
        dst_path = skip_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def write_skip_log(
    row: pd.Series,
    reason: str,
    skipped_csv: Path,
    copied_path: str = ""
) -> None:
    ensure_dir(skipped_csv.parent)

    log_row = pd.DataFrame(
        [{
            "image_name": row.get("image_name", ""),
            "path": row.get("path", ""),
            "data": row.get("data", ""),
            "label": row.get("label", ""),
            "reason": reason,
            "copied_path": copied_path,
        }]
    )

    if skipped_csv.exists():
        log_row.to_csv(skipped_csv, mode="a", header=False, index=False, encoding="utf-8")
    else:
        log_row.to_csv(skipped_csv, mode="w", header=True, index=False, encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    ensure_dir(PROCESSED_ROOT)
    ensure_dir(IMAGES_DIR)

    # Create skip folders
    ensure_dir(SKIPPED_ROOT)
    ensure_dir(SKIPPED_DIR)

    df_btxrd = load_csv(BTXRD_CSV)
    df_fracatlas = load_csv(FRACATLAS_CSV)
    df = pd.concat([df_btxrd, df_fracatlas], ignore_index=True)

    df["class"] = df["label"].map(assign_class)
    df = df[df["class"].notna()].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No rows remain after filtering by ID_LABELS and OOD_LABELS.")

    manifest_rows = []
    source_counters = {source_name: 1 for source_name in SOURCE_CODE_MAP.keys()}

    skipped_unknown_source = 0
    skipped_missing_image = 0
    skipped_invalid_ext = 0
    skipped_quality_filter = 0

    for _, row in df.iterrows():
        source_dataset = row["data"]
        if source_dataset not in SOURCE_CODE_MAP:
            skipped_unknown_source += 1
            print(f"[WARN] Skip unknown dataset source: {source_dataset}")

            # Copy image to skip folder
            write_skip_log(row, "unknown_source", SKIPPED_CSV, copied_path="")

            continue

        src_path = Path(row["path"])
        if not src_path.exists():
            skipped_missing_image += 1
            print(f"[WARN] Missing image: {src_path}")

            # Copy image to skip folder
            write_skip_log(row, "missing_image", SKIPPED_CSV, copied_path="")

            continue

        if src_path.suffix.lower() not in VALID_EXTS:
            skipped_invalid_ext += 1
            print(f"[WARN] Invalid extension: {src_path}")

            # Copy image to skip folder
            copied_path = copy_to_skip_folder(src_path, SKIPPED_DIR)
            write_skip_log(row, "invalid_extension  ", SKIPPED_CSV, copied_path=copied_path)

            continue

        img = read_image(src_path)
        if img is None:
            skipped_missing_image += 1
            print(f"[WARN] Cannot read image: {src_path}")

            # Copy image to skip folder
            copied_path = copy_to_skip_folder(src_path, SKIPPED_DIR)
            write_skip_log(row, "cannot_read", SKIPPED_CSV, copied_path=copied_path)

            continue

        # Resize NGAY SAU KHI DOC
        img = resize_keep_output(img, SAVE_SIZE)

        if not passes_xray_filter(img):
            skipped_quality_filter += 1

            # Copy image to skip folder
            copied_path = copy_to_skip_folder(src_path, SKIPPED_DIR)
            write_skip_log(row, "quality_filter", SKIPPED_CSV, copied_path=copied_path)

            continue

        idx = source_counters[source_dataset]
        base_name_id = build_base_name_id(source_dataset, idx)
        source_counters[source_dataset] += 1

        variants = generate_variants(img)

        for variant_name, variant_img in variants:
            diversity_code = DIVERSITY_CODES[variant_name]
            final_name_id = f"{base_name_id}{diversity_code}"

            processed_img = variant_img
            out_path = IMAGES_DIR / f"{final_name_id}.png"
            save_image(processed_img, out_path)

            manifest_rows.append(
                {
                    "name_id": final_name_id,
                    "data": source_dataset,
                    "path": str(out_path),
                    "label": row["label"],
                    "class": row["class"],
                }
            )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "name_id",
            "data",
            "path",
            "label",
            "class",
        ],
    )

    if manifest.empty:
        raise RuntimeError("No valid rows written to manifest.")

    manifest.to_csv(MANIFEST_CSV, index=False, encoding="utf-8")

    print(f"[OK] wrote manifest: {MANIFEST_CSV}")
    print(f"[INFO] total rows: {len(manifest)}")
    print(f"[INFO] skipped_unknown_source: {skipped_unknown_source}")
    print(f"[INFO] skipped_missing_image: {skipped_missing_image}")
    print(f"[INFO] skipped_invalid_ext: {skipped_invalid_ext}")
    print(f"[INFO] skipped_quality_filter: {skipped_quality_filter}")

    print("\n[FINAL SUMMARY]")
    print(f"Final images after augmentation: {len(manifest)}")
    print("\nBy class:")
    print(manifest["class"].value_counts())
    print("\nBy label:")
    print(manifest["label"].value_counts())
    print("\nBy dataset:")
    print(manifest["data"].value_counts())


if __name__ == "__main__":
    main()