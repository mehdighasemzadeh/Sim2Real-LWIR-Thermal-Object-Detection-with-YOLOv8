# Dual-Stage Thermal Vision YOLOv8 Pipeline

An end-to-end **Sim2Real thermal object detection pipeline** using **YOLOv8 Nano** and **YOLOv8 Small**.

This project bridges the domain gap between visible (RGB) and Long-Wave Infrared (LWIR) thermal imagery through a **two-stage training strategy**:

**COCO RGB → Synthetic LWIR → Real FLIR Thermal Fine-Tuning**

The resulting models are designed with both **computer-vision performance and edge deployment** in mind, including ONNX Runtime, TensorRT, OpenVINO, and embedded AI platforms.

---

## 🎥 Demo

The following video demonstrates the final thermal object detection models on real **FLIR ADAS thermal imagery**.

**YOLOv8 Nano and YOLOv8 Small — Stage 2 real-thermal fine-tuned models**

<p align="center">
  <video src="./videos/test.gif" controls width="900">
    Your browser does not support embedded videos.
    <a href="./videos/test.mp4">View the demo video</a>.
  </video>
</p>

> **Demo:** Real LWIR thermal detection after Sim2Real training and fine-tuning on the FLIR ADAS dataset.

### Results at a glance

| Model | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8 Nano** | **74.8%** | **76.2%** | **83.7%** | **59.3%** |
| **YOLOv8 Small** | **77.7%** | **76.8%** | **85.4%** | **63.6%** |

The demo video shows the models operating directly on **real thermal imagery**, rather than synthetic images.

---

## 🔥 Sim2Real Pipeline

The core idea of this project is to reduce the RGB-to-LWIR domain gap before fine-tuning on real thermal data.

```text
                    COCO 2017
                       │
                       ▼
              ┌─────────────────┐
              │ RGB → Synthetic │
              │      LWIR       │
              └────────┬────────┘
                       │
                       ▼
              Stage 1 Pre-training
              YOLOv8 Nano / Small
                       │
                       ▼
                Synthetic LWIR
                       │
                       │
                       ▼
              ┌─────────────────┐
              │  Real FLIR ADAS │
              │     Thermal     │
              └────────┬────────┘
                       │
                       ▼
              Stage 2 Fine-tuning
              YOLOv8 Nano / Small
                       │
                       ▼
                Final Thermal
               Detection Models
```

---

## Key Features

- **5 Target Classes:** `person`, `car`, `motorcycle`, `train`, `truck`.
- **Stage 1 — Synthetic Domain Pre-training:** Converts RGB COCO 2017 images into synthetic LWIR imagery using Albumentations, including grayscale conversion, thermal polarity inversion, noise, posterization, and blur.
- **Stage 2 — Real Thermal Fine-Tuning:** Converts local FLIR ADAS COCO annotations to YOLO format, applies rare-class oversampling (`train`, `motorcycle`, `truck`), and fine-tunes on real thermal imagery.
- **Thermal Hyperparameter Tuning:** Disables color-space jittering (`hsv_h=0`, `hsv_s=0`, `hsv_v=0`) to better match single-channel and grayscale thermal signals.
- **Edge Deployment Ready:** Automatically exports models to simplified ONNX format (`opset=12`, static shapes).
- **Two Model Sizes:** YOLOv8 Nano for lightweight inference and YOLOv8 Small for higher detection accuracy.
- **Real-World Thermal Evaluation:** Final models are evaluated on real FLIR ADAS thermal imagery.

---

## Repository Structure

```text
.
├── README.md
├── FLIR_DATASET_GUIDE.md
├── stage_1_training_script.py
├── stage_2_training_script.py
├── videos/
│   └── test.mp4                         # Final thermal detection demo
└── datasets/                            # Local dataset root (excluded from Git)
    ├── download_coco2017.py
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
cd datasets
python download_coco2017.py
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
python stage_1_training_script.py \
   --coco-root ./datasets/coco \
   --output-dir ./output/coco_5cls_lwir_aerial \
   --epochs 200 \
   --polarity white
```

**Output:**

Trained PyTorch models will be saved to:

```text
./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/
```

### Step 2: Stage 2 — Data Preparation & Real FLIR Fine-Tuning

#### Option A: Run Data Preparation Only

If you only want to parse COCO JSON annotations, apply class mapping, and generate YOLO `.txt` labels without starting model training:

```bash
python stage_2_training_script.py \
   --flir-root ./datasets/FLIR_ADAS \
   --output-dir ./output/flir_5cls_yolov8_finetuned \
   --prep-only
```

#### Option B: Full Stage 2 Fine-Tuning & ONNX Export

Run full data conversion, oversampling, and Stage 2 fine-tuning using the Stage 1 pretrained weights:

```bash
python stage_2_training_script.py \
   --flir-root ./datasets/FLIR_ADAS \
   --output-dir ./output/flir_5cls_yolov8_finetuned \
   --weights-nano ./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/yolov8n_lwir_aerial_5cls/weights/best.pt \
   --weights-small ./output/coco_5cls_lwir_aerial/yolov8_thermal_quadcopter/yolov8s_lwir_aerial_5cls/weights/best.pt \
   --epochs 150
```

---

## Command Line Arguments Reference

### `stage_1_training_script.py` — Stage 1

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--coco-root` | `str` | `./datasets/coco` | Path to COCO dataset root directory |
| `--output-dir` | `str` | `./output/coco_5cls_lwir_aerial` | Target output directory |
| `--epochs` | `int` | `200` | Number of training epochs |
| `--batch-size` | `int` | Auto (`16`/`8`) | Custom batch size |
| `--polarity` | `str` | `white` | Thermal polarity (`white` for White-Hot, `black` for Black-Hot) |
| `--force-rebuild` | flag | `False` | Force reprocessing of transformed images |

### `stage_2_training_script.py` — Stage 2

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

### Improvement Through the Pipeline

The results demonstrate the effect of each training stage:

```text
YOLOv8-Nano

Stock RGB        mAP50 = 19.8%
       │
       ▼
Synthetic LWIR   mAP50 = 37.1%
       │
       ▼
Real FLIR        mAP50 = 83.7%


YOLOv8-Small

Stock RGB        mAP50 = 21.3%
       │
       ▼
Synthetic LWIR   mAP50 = 39.6%
       │
       ▼
Real FLIR        mAP50 = 85.4%
```

---

## Model Export & Deployment

The fine-tuning scripts automatically export trained PyTorch checkpoints (`best.pt`) to ONNX format with settings optimized for standard inference engines:

- `opset=12`
- `simplify=True`
- `dynamic=False`

The resulting `.onnx` model can be deployed with:

- **ONNX Runtime**
- **TensorRT**
- **OpenVINO**
- Custom embedded/edge AI accelerators
- CPU or GPU inference platforms

---

## Project Summary

This project demonstrates a practical **Sim2Real approach for thermal object detection**:

> **RGB data → Synthetic LWIR pre-training → Real LWIR fine-tuning → Edge-ready object detection**

The final models achieve **83.7% mAP50 with YOLOv8 Nano** and **85.4% mAP50 with YOLOv8 Small** on the FLIR thermal validation set, while retaining lightweight architectures suitable for real-time and embedded computer-vision applications.
