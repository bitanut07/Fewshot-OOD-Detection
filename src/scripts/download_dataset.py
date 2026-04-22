import re
import sys
import yaml
import requests
from pathlib import Path
import zipfile
import shutil


API_BASE = "https://api.figshare.com/v2"


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def extract_article_id(article_url: str) -> int:
    clean_url = article_url.split("?")[0].rstrip("/")
    last = clean_url.split("/")[-1]

    if not last.isdigit():
        raise ValueError(f"Khong parse duoc article_id tu URL: {article_url}")

    return int(last)


def get_article_files(article_id: int):
    url = f"{API_BASE}/articles/{article_id}/files"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def choose_target_file(files, expected_files):
    if not files:
        raise FileNotFoundError("Khong tim thay file nao trong article.")

    if not expected_files:
        return files[0]

    expected_lower = [x.lower() for x in expected_files]
    for f in files:
        if f["name"].lower() in expected_lower:
            return f

    available = [f["name"] for f in files]
    raise FileNotFoundError(
        f"Khong tim thay file khop expected_files={expected_files}. "
        f"Available files: {available}"
    )


def download_file(download_url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] url = {download_url}")
    print(f"[DOWNLOAD] output_path = {output_path}")
    print("[DOWNLOAD] sending request...")

    with requests.get(download_url, stream=True, timeout=(30, 300)) as response:
        print("[DOWNLOAD] response received")
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
        print(f"[DOWNLOAD] content_length = {content_length}")

        total = 0
        with open(output_path, "wb") as f:
            for idx, chunk in enumerate(response.iter_content(chunk_size=1024 * 1024), 1):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

                    if idx <= 5 or idx % 10 == 0:
                        if content_length:
                            percent = total * 100 / content_length
                            print(f"[DOWNLOAD] {total}/{content_length} bytes ({percent:.2f}%)")
                        else:
                            print(f"[DOWNLOAD] downloaded {total} bytes")

    print(f"[DOWNLOAD] completed: {output_path} ({total} bytes)")


def unzip_file(zip_path: Path, extract_to: Path):
    print(f"[UNZIP] zip_path = {zip_path}")
    print(f"[UNZIP] extract_to = {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)
    print("[UNZIP] extract_to ensured")

    print("[UNZIP] opening zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("[UNZIP] zip opened")

        print("[UNZIP] reading members...")
        members = zip_ref.infolist()
        total = len(members)
        print(f"[UNZIP] total members = {total}")

        for idx, member in enumerate(members, 1):
            zip_ref.extract(member, extract_to)

            if idx <= 5 or idx % 100 == 0 or idx == total:
                print(f"[UNZIP] Extracted {idx}/{total}: {member.filename}")

    print(f"[UNZIP] Extracted to: {extract_to}")


def _is_safe_to_flatten(parent: Path, child: Path) -> bool:
    """Only flatten when nested folder is a simple wrapper directory."""
    if not child.is_dir():
        return False

    child_items = list(child.iterdir())
    parent_items = [p for p in parent.iterdir() if p != child]
    child_names = {p.name for p in child_items}
    parent_names = {p.name for p in parent_items}

    # Avoid accidental overwrite when moving files up one level.
    return len(child_items) > 0 and child_names.isdisjoint(parent_names)


def flatten_single_outer_folder(extract_to: Path):
    """
    If zip extracted into a single wrapper folder, move its contents up.
    Example: data/raw/BTXRD/BTXRD/{images,Annotations,dataset.csv}
          -> data/raw/BTXRD/{images,Annotations,dataset.csv}
    """
    items = [p for p in extract_to.iterdir()]
    if len(items) != 1:
        print("[CLEANUP] Skip flatten: not a single outer folder.")
        return

    outer = items[0]
    if not _is_safe_to_flatten(extract_to, outer):
        print("[CLEANUP] Skip flatten: unsafe or nothing to move.")
        return

    print(f"[CLEANUP] Flatten outer folder: {outer}")
    for child in outer.iterdir():
        target = extract_to / child.name
        shutil.move(str(child), str(target))
    outer.rmdir()
    print("[CLEANUP] Outer folder removed.")


def remove_zip_if_unused(zip_path: Path):
    if not zip_path.exists():
        print("[CLEANUP] Zip already removed.")
        return

    zip_path.unlink()
    print(f"[CLEANUP] Removed zip file: {zip_path}")


def _copy_path(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def download_from_kaggle(dataset_cfg: dict):
    dataset_slug = dataset_cfg.get("dataset_slug")
    if not dataset_slug:
        raise ValueError("Kaggle source can 'dataset_slug' trong config.")

    output_dir = Path(dataset_cfg["output_dir"])
    kaggle_subpath = dataset_cfg.get("kaggle_subpath")

    try:
        import kagglehub  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Chua cai kagglehub. Hay cai bang: pip install kagglehub"
        ) from e

    print(f"[KAGGLE] dataset_slug = {dataset_slug}")
    download_root = Path(kagglehub.dataset_download(dataset_slug))
    print(f"[KAGGLE] downloaded to cache: {download_root}")

    source_path = download_root
    if kaggle_subpath:
        source_path = download_root / kaggle_subpath
        if not source_path.exists():
            raise FileNotFoundError(
                f"Kaggle subpath khong ton tai: {source_path}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[KAGGLE] copying data -> {output_dir}")

    if source_path.is_dir():
        for item in source_path.iterdir():
            _copy_path(item, output_dir / item.name)
    else:
        _copy_path(source_path, output_dir / source_path.name)

    print("[KAGGLE] copy completed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_dataset.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_yaml_config(config_path)
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError(
            "Config khong hop le: can key top-level 'dataset'. "
            "Kiem tra lai file YAML (co the dang bi comment sai indent)."
        )

    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", "").lower()

    if source == "figshare":
        article_url = dataset_cfg["article_url"]
        expected_files = dataset_cfg.get("expected_files", [])
        output_dir = Path(dataset_cfg["output_dir"])

        article_id = extract_article_id(article_url)
        print(f"Article ID: {article_id}")

        files = get_article_files(article_id)
        print("Files in article:")
        for f in files:
            print(f"  - {f['name']} (id={f['id']})")

        target_file = choose_target_file(files, expected_files)
        print(f"Selected file: {target_file['name']}")

        download_url = target_file["download_url"]
        output_path = output_dir / target_file["name"]

        download_file(download_url, output_path)

        if output_path.suffix.lower() == ".zip":
            unzip_dir = output_dir
            unzip_dir.mkdir(parents=True, exist_ok=True)
            unzip_file(output_path, unzip_dir)
            flatten_single_outer_folder(unzip_dir)
            remove_zip_if_unused(output_path)
    elif source == "kaggle":
        download_from_kaggle(dataset_cfg)
    else:
        raise ValueError(
            f"Source '{source}' chua duoc ho tro. Ho tro: figshare | kaggle."
        )

    print("Done.")


if __name__ == "__main__":
    main()