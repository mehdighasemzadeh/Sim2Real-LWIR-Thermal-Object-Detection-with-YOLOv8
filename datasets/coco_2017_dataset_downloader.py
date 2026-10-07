"""
COCO 2017 Dataset Automated Downloader & Extractor

Downloads and extracts:
- train2017.zip (118,287 images, ~18 GB)
- val2017.zip (5,000 images, ~1 GB)
- annotations_trainval2017.zip (241 MB)

Usage:
    python download_coco2017.py --output-dir ./datasets/coco
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path


COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip"
}


class DownloadProgressBar:
    """Displays a download progress bar in stdout."""
    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        if percent != self.last_percent and percent <= 100:
            self.last_percent = percent
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f"\r  [{bar}] {percent}% ({mb_downloaded:.1f} / {mb_total:.1f} MB) - {self.filename}")
            sys.stdout.flush()
            if percent == 100:
                print()


def download_and_extract(name: str, url: str, output_dir: Path, keep_zips: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{name}.zip"

    # Check if target folder/file already extracted
    if name == "annotations" and (output_dir / "annotations" / "instances_train2017.json").exists():
        print(f"[Skipped] {name} annotations already exist.")
        return
    elif name in ["train2017", "val2017"] and (output_dir / name).exists():
        if len(list((output_dir / name).glob("*.jpg"))) > 100:
            print(f"[Skipped] {name} images already extracted.")
            return

    # Download file if zip does not exist
    if not zip_path.exists():
        print(f"\nDownloading {name} from {url}...")
        try:
            progress_callback = DownloadProgressBar(f"{name}.zip")
            urllib.request.urlretrieve(url, zip_path, reporthook=progress_callback)
        except Exception as e:
            print(f"\n[Error] Failed to download {name}: {e}")
            if zip_path.exists():
                zip_path.unlink()
            sys.exit(1)

    # Extract zip file
    print(f"Extracting {zip_path.name} to {output_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"[Success] Extracted {name}.")
    except Exception as e:
        print(f"[Error] Failed to extract {zip_path.name}: {e}")
        sys.exit(1)

    # Clean up zip file
    if not keep_zips and zip_path.exists():
        os.remove(zip_path)
        print(f"Removed archive {zip_path.name}.")


def main():
    parser = argparse.ArgumentParser(description="Download and extract MS COCO 2017 dataset")
    parser.add_argument("--output-dir", type=str, default="./coco", help="Target output directory")
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip archives after extraction")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    print("=" * 60)
    print(f"COCO 2017 DATASET DOWNLOADER")
    print(f"Destination: {out_path}")
    print("=" * 60)

    # Download annotations first (small), then val, then train
    download_and_extract("annotations", COCO_URLS["annotations"], out_path, keep_zips=args.keep_zips)
    download_and_extract("val2017", COCO_URLS["val2017"], out_path, keep_zips=args.keep_zips)
    download_and_extract("train2017", COCO_URLS["train2017"], keep_zips=args.keep_zips, output_dir=out_path)

    print("\n" + "=" * 60)
    print("[Complete] COCO 2017 dataset downloaded and ready for Stage 1 training!")
    print("=" * 60)


if __name__ == "__main__":
    main()