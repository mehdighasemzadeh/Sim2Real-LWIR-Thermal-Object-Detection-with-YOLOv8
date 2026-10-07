# Dual-Stage Thermal Vision YOLOv8 Pipeline

This repository provides an end-to-end framework for domain-adapted thermal object detection using **YOLOv8 Nano** and **YOLOv8 Small**. 

The architecture bridges the domain gap between visible (RGB) and Long-Wave Infrared (LWIR) thermal spectrums using a two-stage training strategy optimized for general-purpose edge and workstation deployment (e.g., TensorRT, ONNX Runtime, OpenVINO, or custom accelerators).

---

## Key Features

- **5 Target Classes:** `person`, `car`, `motorcycle`, `train`, `truck`.
- **Stage 1 (Synthetic Domain Pre-training):** Converts RGB COCO 2017 images into synthetic LWIR imagery using Albumentations (grayscale conversion, thermal polarity inversion, noise, posterization, and blur).
- **Stage 2 (Real Thermal Fine-Tuning):** Converts local FLIR ADAS COCO annotations to YOLO format, applies rare class oversampling (`train`, `motorcycle`, `truck`), and fine-tunes on real thermal imagery.
- **Thermal Hyperparameter Tuning:** Disables color-space jittering (`hsv_h=0`, `hsv_s=0`, `hsv_v=0`) to optimize loss convergence for single-channel and grayscale thermal signals.
- **Cross-Platform Deployment Ready:** Automated export to simplified ONNX format (`opset=12`, static shapes) ready for deployment across edge GPUs, CPUs, or embedded microprocessors.

---

## Repository Structure

```text
.
├── README.md                                # Comprehensive documentation
├── FLIR_DATASET_GUIDE.md                    # Guide for manual FLIR dataset setup
├── stage_1_training_script.py               # Stage 1: Synthetic LWIR pre-training
├── stage_2_training_script.py               # Stage 2: Data prep & real thermal fine-tuning
└── datasets/                                # Local dataset root (excluded from Git)
    download_coco2017.py                     # COCO 2017 downloader & extractor
    ├── coco/
    │   ├── annotations/
    │   ├── train2017/
    │   └── val2017/
    └── FLIR_ADAS/
        ├── coco.json
        └── data/
```

---

## Environment Setup

### Requirements

Ensure you have Python 3.8+ and PyTorch installed.

```bash
pip install ultralytics albumentations pycocotools torch torchvision opencv-python numpy pyyaml
```

---

## Dataset Acquisition & Layout

### Stage 1: COCO 2017 Dataset
You can automatically download and extract the dataset using the included script:

```bash
python download_coco2017.py --output-dir ./datasets/coco
```

Or manually extract COCO 2017 so that the directory matches:
```text
./datasets/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

### Stage 2: FLIR ADAS Thermal Dataset
Follow the step-by-step instructions in [`FLIR_DATASET_GUIDE.md`](./FLIR_DATASET_GUIDE.md) to download and extract the archive into:

```text
./datasets/FLIR_ADAS/
├── coco.json
└── data/
    ├── FLIR_00001.jpg
    ├── FLIR_00002.jpg
    └── ...
```

---

## Pipeline Execution Guide

### Step 1: Stage 1 — Synthetic LWIR Pre-Training

Train YOLOv8 Nano and Small on COCO 2017 with dynamic synthetic LWIR augmentations:

```bash
python yolov8_5_class_lwir_fine_tuning_v2.py \
    --coco-root ./datasets/coco \
    --output-dir ./output/coco_5cls_lwir_aerial \
    --epochs 200 \
    --polarity white
```

*Output:* Trained PyTorch models will be saved to `./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/`.

### Step 2: Stage 2 — Data Preparation & Real FLIR Fine-Tuning

#### Option A: Run Data Preparation Only
If you only want to parse COCO JSON annotations, apply class mapping, and generate YOLO `.txt` labels without starting model training:

```bash
python flir_real_thermal_yolov8_fine_tuning.py \
    --flir-root ./datasets/FLIR_ADAS \
    --output-dir ./output/flir_5cls_yolov8_finetuned \
    --prep-only
```

#### Option B: Full Stage 2 Fine-Tuning & ONNX Export
Run full data conversion, oversampling, and Stage 2 fine-tuning using the Stage 1 pretrained weights:

```bash
python flir_real_thermal_yolov8_fine_tuning.py \
    --flir-root ./datasets/FLIR_ADAS \
    --output-dir ./output/flir_5cls_yolov8_finetuned \
    --weights-nano ./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/yolov8n_lwir_aerial_5cls/weights/best.pt \
    --weights-small ./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/yolov8s_lwir_aerial_5cls/weights/best.pt \
    --epochs 150
```

---

## Command Line Arguments Reference

### `yolov8_5_class_lwir_fine_tuning_v2.py` (Stage 1)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--coco-root` | `str` | `./datasets/coco` | Path to COCO dataset root directory |
| `--output-dir` | `str` | `./output/coco_5cls_lwir_aerial` | Target output directory |
| `--epochs` | `int` | `200` | Number of training epochs |
| `--batch-size` | `int` | Auto (`16`/`8`) | Custom batch size |
| `--polarity` | `str` | `white` | Thermal polarity (`white` for White-Hot, `black` for Black-Hot) |
| `--force-rebuild` | flag | `False` | Force reprocessing of transformed images |

### `flir_real_thermal_yolov8_fine_tuning.py` (Stage 2)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--flir-root` | `str` | `./datasets/FLIR_ADAS` | Path to FLIR ADAS dataset directory |
| `--output-dir` | `str` | `./output/flir_5cls_yolov8_finetuned` | Destination folder for processed dataset and runs |
| `--weights-nano` | `str` | `""` | Optional Stage 1 `best.pt` path for YOLOv8 Nano |
| `--weights-small` | `str` | `""` | Optional Stage 1 `best.pt` path for YOLOv8 Small |
| `--epochs` | `int` | `150` | Fine-tuning epoch count |
| `--prep-only` | flag | `False` | Perform COCO-to-YOLO preparation only without training |
| `--force-rebuild` | flag | `False` | Force dataset regeneration |

---

## Benchmark Results

Performance comparison across pipeline stages evaluated on FLIR thermal validation data:

| Stage / Model | Architecture | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 0: Stock Pretrained** | YOLOv8-Nano | 27.2% | 11.5% | 19.8% | 12.8% |
| **Stage 1: Synthetic LWIR** | YOLOv8-Nano | 53.3% | 23.6% | 37.1% | 22.8% |
| **Stage 2: Real Thermal Fine-Tuned** | **YOLOv8-Nano** | **74.8%** | **76.2%** | **83.7%** | **59.3%** |
| **Stage 0: Stock Pretrained** | YOLOv8-Small | 27.6% | 13.9% | 21.3% | 14.3% |
| **Stage 1: Synthetic LWIR** | YOLOv8-Small | 54.2% | 28.8% | 39.6% | 25.7% |
| **Stage 2: Real Thermal Fine-Tuned** | **YOLOv8-Small** | **77.7%** | **76.8%** | **85.4%** | **63.6%** |

---

## Model Export & Deployment

The fine-tuning scripts automatically export trained PyTorch checkpoints (`best.pt`) to ONNX format with settings optimized for standard inference engines:
- `opset=12`
- `simplify=True`
- Dynamic shapes disabled (`dynamic=False`)

The resulting `.onnx` model can be deployed directly with **ONNX Runtime**, **TensorRT**, **OpenVINO**, or converted to proprietary edge hardware format using your preferred vendor compiler toolchain.
