"""
YOLOv8 Stage 1: COCO 5-Class Synthetic LWIR Transformation & Training

This script prepares the COCO 2017 dataset by filtering for 5 target classes,
applying dynamic Albumentations thermal transformations (grayscale, inversion,
contrast, noise, blur), oversampling underrepresented target classes, and training
YOLOv8 Nano and Small architectures for aerial thermal vision tasks.

Usage:
    python yolov8_5_class_lwir_fine_tuning_v2.py --coco-root ./datasets/coco --epochs 200
"""

import os
import sys
import shutil
import yaml
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO

# Global Default Configurations
CLASSES: List[str] = ['person', 'car', 'motorcycle', 'train', 'truck']
RARE_CLASSES: List[str] = ['train', 'motorcycle', 'truck']
RARE_REPEAT: int = 3
SEED: int = 0


def check_gpu_environment() -> None:
    """Prints GPU diagnostic information and VRAM capacity warning if low."""
    print("=" * 60)
    print("STAGE 1: ENVIRONMENT & HARDWARE CHECK")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" Detected GPU : {gpu_name}")
        print(f" VRAM Available: {vram_gb:.2f} GB")
        if vram_gb < 7.5:
            print(" [Warning] VRAM < 8GB detected. Lowering default batch sizes recommended.")
    else:
        print(" [Warning] No CUDA GPU detected. Training will run on CPU.")
    print("=" * 60)


def get_lwir_transforms(is_val: bool = False, polarity: str = "white"):
    """
    Constructs Albumentations transform pipeline for LWIR/thermal image simulation.

    Args:
        is_val: If True, uses deterministic transforms for validation reproducibility.
        polarity: 'white' for White-Hot thermal, 'black' for Black-Hot thermal.
    """
    try:
        import albumentations as A
    except ImportError as e:
        raise ImportError("Albumentations library is required. Install with: pip install albumentations") from e

    invert_p = 0.5 if not is_val else (1.0 if polarity == "black" else 0.0)

    if is_val:
        return A.Compose([
            A.ToGray(p=1.0),
            A.InvertImg(p=invert_p),
            A.Posterize(num_bits=4, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ])

    return A.Compose([
        A.ToGray(p=1.0),
        A.InvertImg(p=invert_p),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(60, 150), p=0.5),
        A.Posterize(num_bits=4, p=0.3),
        A.Blur(blur_limit=5, p=0.3),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(60, 90), p=0.2),
    ])


def _apply_lwir_to_file(src_path: Path, dst_path: Path, transform) -> None:
    """Reads source image, applies synthetic thermal transformations, and writes to destination."""
    img = cv2.imread(str(src_path))
    if img is None:
        shutil.copy(src_path, dst_path)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)["image"]
    out_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), out_bgr)


def prepare_coco_dataset(
    coco_dir: str,
    output_dir: str,
    target_classes: List[str],
    polarity: str = "white",
    force_rebuild: bool = False
) -> None:
    """Extracts target classes from COCO 2017, converts images to synthetic LWIR, and outputs YOLO labels."""
    print("\n" + "=" * 60)
    print("PREPARING COCO DATASET (SYNTHETIC LWIR TRANSFORMATION)")
    print("=" * 60)

    coco_path = Path(coco_dir)
    out_path = Path(output_dir)

    done_marker = out_path / ".prepared"
    if done_marker.exists() and not force_rebuild:
        print(f"[Skipped] Dataset already prepared at '{output_dir}'. Use --force-rebuild to regenerate.")
        return

    if force_rebuild and out_path.exists():
        print(f"[Rebuild] Purging existing dataset directory at {out_path}...")
        shutil.rmtree(out_path)

    train_tf = get_lwir_transforms(is_val=False, polarity=polarity)
    val_tf = get_lwir_transforms(is_val=True, polarity=polarity)

    for split in ["train2017", "val2017"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = [
        ("train2017", "instances_train2017.json"),
        ("val2017", "instances_val2017.json")
    ]

    for split_name, ann_filename in splits:
        ann_path = coco_path / "annotations" / ann_filename
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file: {ann_path}\n"
                f"Ensure COCO dataset is extracted properly in '{coco_path}'."
            )

        print(f" Processing {split_name}...")
        coco = COCO(str(ann_path))

        cat_id_to_yolo: Dict[int, int] = {}
        for idx, name in enumerate(target_classes):
            cat_ids = coco.getCatIds(catNms=[name])
            if cat_ids:
                cat_id_to_yolo[cat_ids[0]] = idx

        valid_cat_ids = list(cat_id_to_yolo.keys())

        img_ids = []
        for cid in valid_cat_ids:
            img_ids.extend(coco.getImgIds(catIds=cid))
        img_ids = sorted(list(set(img_ids)))

        tf = val_tf if split_name == "val2017" else train_tf

        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            fname = info["file_name"]
            src_img = coco_path / split_name / fname
            dst_img = out_path / "images" / split_name / fname

            if not src_img.exists() or dst_img.exists():
                continue

            _apply_lwir_to_file(src_img, dst_img, tf)

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=valid_cat_ids, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            img_w, img_h = info["width"], info["height"]

            label_path = out_path / "labels" / split_name / (Path(fname).stem + ".txt")
            valid_boxes = []

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_yolo:
                    continue

                yolo_cls = cat_id_to_yolo[cat_id]
                x, y, w, h = ann["bbox"]

                x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
                y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
                norm_w = max(0.0, min(1.0, w / img_w))
                norm_h = max(0.0, min(1.0, h / img_h))

                if norm_w <= 0.001 or norm_h <= 0.001:
                    continue

                valid_boxes.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(valid_boxes) + ("\n" if valid_boxes else ""))

    done_marker.touch()
    print(" Dataset preparation complete.")


def create_oversampled_train_list(
    output_dir: str,
    target_classes: List[str],
    rare_classes: List[str],
    repeat: int = 3
) -> str:
    """Generates train.txt containing duplicated image paths for rare target classes."""
    out_path = Path(output_dir)
    train_img_dir = out_path / "images" / "train2017"
    train_lbl_dir = out_path / "labels" / "train2017"

    images = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
    if not images:
        raise RuntimeError(f"No training images found in {train_img_dir}")

    rare_class_ids = {target_classes.index(c) for c in rare_classes if c in target_classes}
    rare_images = []

    for img in images:
        lbl_file = train_lbl_dir / (img.stem + ".txt")
        if not lbl_file.exists():
            continue
        with open(lbl_file, "r") as f:
            classes_in_file = {int(line.split()[0]) for line in f if line.strip()}
        if classes_in_file & rare_class_ids:
            rare_images.append(img)

    lines = [str(p.resolve()) for p in images]
    for img in rare_images:
        lines.extend([str(img.resolve())] * (repeat - 1))

    list_path = out_path / "train.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(lines))

    print(f" Oversampled train list generated:")
    print(f"  - Unique train images: {len(images)}")
    print(f"  - Rare class images  : {len(rare_images)}")
    print(f"  - Total list entries : {len(lines)}")
    return str(list_path)


def create_dataset_yaml(output_dir: str, target_classes: List[str]) -> str:
    """Generates dataset definition YAML file required by Ultralytics YOLO."""
    yaml_path = Path(output_dir) / "drone_lwir_5cls.yaml"
    content = {
        "path": str(Path(output_dir).resolve()),
        "train": "train.txt",
        "val": "images/val2017",
        "names": {i: n for i, n in enumerate(target_classes)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f" Dataset YAML created: {yaml_path}")
    return str(yaml_path)


def train_yolov8_model(
    model_name: str,
    weights_path: str,
    data_yaml: str,
    epochs: int = 200,
    batch_size: Optional[int] = None
) -> str:
    """Trains a YOLOv8 architecture on the synthetic LWIR dataset with thermal hyperparameter tuning."""
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()} (INITIAL WEIGHTS: {weights_path})")
    print("=" * 60)

    model = YOLO(weights_path)

    if batch_size is None:
        batch_size = 8 if model_name == "yolov8s" else 16

    train_params = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch_size,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "amp": True,
        "project": "yolov8_thermal_quadcopter",
        "name": f"{model_name}_lwir_aerial_5cls",
        "save": True,
        "patience": 30,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "cos_lr": True,
        "seed": SEED,

        # Thermal image tuning (disable color jitter)
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,

        # Spatial / Geometric augmentations
        "degrees": 30.0,
        "translate": 0.15,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.1,
        "close_mosaic": 15,
        "erasing": 0.10,

        # Loss weights
        "cls": 0.8,
        "cache": False,
    }

    results = model.train(**train_params)
    return str(model.trainer.best)


def export_for_npu(model_path: str) -> None:
    """Exports trained best PyTorch model checkpoint to ONNX format for RK3588 NPU compilation."""
    if not model_path or not Path(model_path).exists():
        print(f" Export skipped: model checkpoint not found ({model_path}).")
        return

    print("\n" + "=" * 60)
    print(f"EXPORTING ONNX FOR RK3588 NPU: {model_path}")
    print("=" * 60)

    model = YOLO(model_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f" ONNX export successful: {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: YOLOv8 COCO LWIR Synthetic Pretraining")
    parser.add_argument("--coco-root", type=str, default="./datasets/coco", help="Path to COCO dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./output/coco_5cls_lwir_aerial", help="Target output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: auto per architecture)")
    parser.add_argument("--polarity", type=str, default="white", choices=["white", "black"], help="Thermal polarity")
    parser.add_argument("--weights-nano", type=str, default="yolov8n.pt", help="Initial weights for YOLOv8 Nano")
    parser.add_argument("--weights-small", type=str, default="yolov8s.pt", help="Initial weights for YOLOv8 Small")
    parser.add_argument("--force-rebuild", action="store_true", help="Force dataset processing rebuild")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    check_gpu_environment()

    # Step 1: Prepare synthetic LWIR COCO dataset
    prepare_coco_dataset(
        coco_dir=args.coco_root,
        output_dir=args.output_dir,
        target_classes=CLASSES,
        polarity=args.polarity,
        force_rebuild=args.force_rebuild,
    )

    # Step 2: Build oversampled train list
    create_oversampled_train_list(
        output_dir=args.output_dir,
        target_classes=CLASSES,
        rare_classes=RARE_CLASSES,
        repeat=RARE_REPEAT,
    )

    # Step 3: Write dataset YAML
    yaml_path = create_dataset_yaml(output_dir=args.output_dir, target_classes=CLASSES)

    # Step 4: Execute training for Nano and Small architectures
    models_to_train = [
        {"name": "yolov8n", "weights": args.weights_nano},
        {"name": "yolov8s", "weights": args.weights_small},
    ]

    for item in models_to_train:
        name = item["name"]
        weights = item["weights"]

        if not Path(weights).exists() and not weights.endswith(".pt"):
            print(f"\n[Notice] Specified weights '{weights}' not found. Falling back to default '{name}.pt'")
            weights = f"{name}.pt"

        best_ckpt = train_yolov8_model(
            model_name=name,
            weights_path=weights,
            data_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        export_for_npu(best_ckpt)

    print("\n" + "=" * 60)
    print("[Stage 1 Complete] Synthetic LWIR training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()