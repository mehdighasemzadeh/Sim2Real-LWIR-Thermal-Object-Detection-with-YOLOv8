"""
FLIR Real Thermal Data Preparation & YOLOv8 Fine-Tuning Pipeline

This script handles dataset preparation and Stage 2 fine-tuning for real FLIR thermal data.
It assumes the dataset has been downloaded manually and placed in the target directory.

Data Preparation Steps:
1. Parses FLIR COCO JSON annotations (`coco.json`).
2. Maps FLIR categories into 5 target classes: ['person', 'car', 'motorcycle', 'train', 'truck'].
3. Creates YOLO format labels (.txt) and splits images into train/val subsets.
4. Generates an oversampled training manifest (`train.txt`) to balance rare target classes.
5. Emits `flir_real_thermal_5cls.yaml` for Ultralytics YOLO training.
6. Fine-tunes YOLOv8 Nano and Small architectures and exports ONNX for RK3588 NPU.

Usage:
    python flir_real_thermal_yolov8_fine_tuning.py \
        --flir-root ./datasets/FLIR_ADAS \
        --output-dir ./output/flir_5cls_yolov8_finetuned
"""

import os
import sys
import shutil
import yaml
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
import ultralytics.utils.checks

# Disable automatic remote checks during validation
ultralytics.utils.checks.check_amp = lambda *args, **kwargs: True

# Global Default Configurations
TARGET_CLASSES: List[str] = ['person', 'car', 'motorcycle', 'train', 'truck']

FLIR_CLASS_MAPPING: Dict[str, str] = {
    'person': 'person',
    'car': 'car',
    'bike': 'motorcycle',
    'motor': 'motorcycle',
    'truck': 'truck',
    'bus': 'truck',
    'train': 'train'
}

RARE_CLASSES: List[str] = ['train', 'motorcycle', 'truck']
RARE_REPEAT: int = 4
SEED: int = 42


def check_gpu_environment() -> None:
    """Prints GPU diagnostic information and VRAM capacity."""
    print("=" * 60)
    print("FLIR DATA PREPARATION & TRAINING: ENVIRONMENT CHECK")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" Detected GPU : {gpu_name}")
        print(f" VRAM Available: {vram_gb:.2f} GB")
        if vram_gb < 7.5:
            print(" [Warning] VRAM < 8GB detected. Lower batch sizes recommended.")
    else:
        print(" [Warning] No CUDA GPU detected. Processing will proceed on CPU.")
    print("=" * 60)


def convert_flir_coco_to_yolo(flir_root: str, output_dir: str, force_rebuild: bool = False) -> None:
    """
    Parses FLIR COCO json annotations and prepares YOLO label files with train/val splits.

    Args:
        flir_root: Directory containing manually downloaded FLIR ADAS files (coco.json and images).
        output_dir: Destination folder for processed dataset.
        force_rebuild: If True, purges and regenerates target output folder.
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION: PARSING FLIR COCO ANNOTATIONS -> YOLO FORMAT")
    print("=" * 60)

    flir_path = Path(flir_root)
    out_path = Path(output_dir)

    # Search for annotation file candidate
    coco_json_path = flir_path / "coco.json"
    if not coco_json_path.exists():
        json_candidates = list(flir_path.rglob("*.json"))
        if json_candidates:
            coco_json_path = json_candidates[0]
            print(f" Found alternate annotation file: {coco_json_path}")
        else:
            raise FileNotFoundError(
                f"\n[Error] Cannot locate COCO annotation file in: {flir_path}\n"
                f"Please follow FLIR_DATASET_GUIDE.md to manually place 'coco.json' into '{flir_path}'."
            )

    done_marker = out_path / ".prepared"
    if done_marker.exists() and not force_rebuild:
        print(f"[Skipped] FLIR dataset already prepared at '{output_dir}'. Use --force-rebuild to refresh.")
        return

    if force_rebuild and out_path.exists():
        print(f"[Rebuild] Purging existing output folder at {out_path}...")
        shutil.rmtree(out_path)

    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    coco = COCO(str(coco_json_path))

    # Build category ID mapping
    cats = coco.loadCats(coco.getCatIds())
    flir_cat_id_to_target: Dict[int, int] = {}

    for c in cats:
        cat_name = c['name'].lower()
        if cat_name in FLIR_CLASS_MAPPING:
            target_class_name = FLIR_CLASS_MAPPING[cat_name]
            target_idx = TARGET_CLASSES.index(target_class_name)
            flir_cat_id_to_target[c['id']] = target_idx
            print(f" Mapped FLIR category '{c['name']}' (ID: {c['id']}) -> Class {target_idx} ('{target_class_name}')")

    img_ids = sorted(coco.getImgIds())
    random.seed(SEED)
    random.shuffle(img_ids)

    # Split dataset: 85% train, 15% validation
    split_idx = int(len(img_ids) * 0.85)
    train_ids = set(img_ids[:split_idx])

    processed_count = 0

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        fname = Path(info["file_name"]).name

        # Locate image in potential subfolders
        src_img = flir_path / "data" / fname
        if not src_img.exists():
            src_img = flir_path / fname
        if not src_img.exists():
            image_search = list(flir_path.rglob(fname))
            if image_search:
                src_img = image_search[0]
            else:
                continue

        split = "train" if img_id in train_ids else "val"
        dst_img = out_path / "images" / split / fname
        dst_lbl = out_path / "labels" / split / (Path(fname).stem + ".txt")

        if not dst_img.exists():
            shutil.copy(src_img, dst_img)

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        img_w, img_h = info["width"], info["height"]

        valid_boxes = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in flir_cat_id_to_target:
                continue

            yolo_class_id = flir_cat_id_to_target[cat_id]
            x, y, w, h = ann["bbox"]

            x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
            y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
            norm_w = max(0.0, min(1.0, w / img_w))
            norm_h = max(0.0, min(1.0, h / img_h))

            if norm_w <= 0.002 or norm_h <= 0.002:
                continue

            valid_boxes.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(dst_lbl, "w") as f:
            f.write("\n".join(valid_boxes) + ("\n" if valid_boxes else ""))

        processed_count += 1

    done_marker.touch()
    print(f" Data Preparation Complete: Converted {processed_count} images and labels.")


def generate_oversampled_list(
    output_dir: str,
    target_classes: List[str],
    rare_classes: List[str],
    repeat: int = 4
) -> str:
    """Generates train.txt containing duplicated image paths for underrepresented rare categories."""
    out_path = Path(output_dir)
    train_img_dir = out_path / "images" / "train"
    train_lbl_dir = out_path / "labels" / "train"

    images = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpeg")))
    if not images:
        raise RuntimeError(f"No training images found in {train_img_dir}")

    rare_ids = {target_classes.index(c) for c in rare_classes if c in target_classes}
    rare_images = []

    for img in images:
        lbl = train_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl, "r") as f:
            classes_in_file = {int(line.split()[0]) for line in f if line.strip()}
        if classes_in_file & rare_ids:
            rare_images.append(img)

    oversampled_lines = [str(p.resolve()) for p in images]
    for img in rare_images:
        oversampled_lines.extend([str(img.resolve())] * (repeat - 1))

    list_path = out_path / "train.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(oversampled_lines))

    print(f" Oversampled Manifest Created:")
    print(f"  - Base Training Images : {len(images)}")
    print(f"  - Rare Class Images    : {len(rare_images)}")
    print(f"  - Manifest List Count  : {len(oversampled_lines)}")
    return str(list_path)


def build_dataset_yaml(output_dir: str, target_classes: List[str]) -> str:
    """Creates the dataset configuration YAML file for YOLOv8 training."""
    yaml_path = Path(output_dir) / "flir_real_thermal_5cls.yaml"
    data = {
        "path": str(Path(output_dir).resolve()),
        "train": "train.txt",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(target_classes)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f" Generated Dataset Config: {yaml_path}")
    return str(yaml_path)


def run_finetuning(
    model_name: str,
    weights_path: str,
    data_yaml: str,
    epochs: int = 150
) -> str:
    """Executes YOLOv8 fine-tuning on the prepared real FLIR thermal dataset."""
    print("\n" + "=" * 60)
    print(f"FINE-TUNING ARCHITECTURE: {model_name.upper()}")
    print("=" * 60)

    resolved_weights = Path(weights_path).resolve() if weights_path else None

    if resolved_weights and resolved_weights.exists():
        print(f" [Weights] Loaded pretrained Stage 1 checkpoint: {resolved_weights}")
        model = YOLO(f"{model_name}.yaml").load(str(resolved_weights))
    else:
        print(f" [Weights] Using baseline pretrained checkpoint '{model_name}.pt'.")
        model = YOLO(f"{model_name}.pt")

    batch_size = 16 if model_name == "yolov8n" else 8

    train_params = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch_size,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "amp": True,
        "project": "yolov8_flir_real_thermal",
        "name": f"{model_name}_flir_finetuned_5cls",
        "save": True,
        "patience": 25,
        "optimizer": "AdamW",
        "lr0": 0.0003,
        "lrf": 0.01,
        "cos_lr": True,
        "seed": SEED,

        # Thermal Hyperparameters (Disable Color Jitter)
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,

        # Geometric Augmentations
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.1,
        "close_mosaic": 10,

        "cls": 1.0,
        "cache": False,
    }

    results = model.train(**train_params)
    return str(model.trainer.best)


def export_onnx_for_rk3588(model_path: str) -> None:
    """Exports trained best checkpoint to ONNX format for RK3588 NPU compilation."""
    if not model_path or not Path(model_path).exists():
        print(f" Export skipped: Checkpoint not found at {model_path}")
        return

    print("\n" + "=" * 60)
    print(f"EXPORTING ONNX FOR RK3588 NPU: {model_path}")
    print("=" * 60)

    model = YOLO(model_path)
    onnx_file = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f" ONNX export successful: {onnx_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="FLIR Real Thermal Data Preparation & Fine-Tuning")
    parser.add_argument("--flir-root", type=str, default="./datasets/FLIR_ADAS", help="Path to manually downloaded FLIR dataset root")
    parser.add_argument("--output-dir", type=str, default="./output/flir_5cls_yolov8_finetuned", help="Target output folder")
    parser.add_argument("--weights-nano", type=str, default="", help="Optional Stage 1 best weights for YOLOv8 Nano")
    parser.add_argument("--weights-small", type=str, default="", help="Optional Stage 1 best weights for YOLOv8 Small")
    parser.add_argument("--epochs", type=int, default=150, help="Number of fine-tuning epochs")
    parser.add_argument("--prep-only", action="store_true", help="Only perform data preparation without starting training")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of converted dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    check_gpu_environment()

    # Step 1: Format FLIR COCO JSON into YOLO format
    convert_flir_coco_to_yolo(args.flir_root, args.output_dir, force_rebuild=args.force_rebuild)

    # Step 2: Generate oversampled train list
    generate_oversampled_list(args.output_dir, TARGET_CLASSES, RARE_CLASSES, repeat=RARE_REPEAT)

    # Step 3: Write dataset YAML file
    yaml_config = build_dataset_yaml(args.output_dir, TARGET_CLASSES)

    if args.prep_only:
        print("\n" + "=" * 60)
        print("[Complete] Data preparation finished successfully (--prep-only flag set).")
        print(f"Dataset YAML ready at: {yaml_config}")
        print("=" * 60)
        return

    # Step 4: Fine-tune Nano and Small models
    models = [
        {"name": "yolov8n", "weights": args.weights_nano},
        {"name": "yolov8s", "weights": args.weights_small}
    ]

    for item in models:
        best_ckpt = run_finetuning(
            model_name=item["name"],
            weights_path=item["weights"],
            data_yaml=yaml_config,
            epochs=args.epochs,
        )
        export_onnx_for_rk3588(best_ckpt)

    print("\n" + "=" * 60)
    print("[Pipeline Complete] FLIR data preparation, fine-tuning, and ONNX export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()