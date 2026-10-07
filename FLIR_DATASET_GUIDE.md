# FLIR ADAS Thermal Dataset: Download & Setup Guide

This guide outlines how to manually obtain the **FLIR ADAS (Thermal)** dataset and organize it for the YOLOv8 thermal vision preparation pipeline.

---

## 1. Where to Download the FLIR ADAS Dataset

Because Teledyne FLIR requires account registration or distribution agreements for direct API downloads, you can manually download the dataset archive from one of the following official or community sources:

1. **Kaggle Dataset Mirror (Recommended):**
   * **URL:** [Kaggle - FLIR Thermal Images Dataset](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
   * **Action:** Click **Download (GB)** to obtain the `.zip` archive.

2. **Official Teledyne FLIR Page:**
   * **URL:** [Teledyne FLIR Free ADAS Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/)
   * **Action:** Complete the registration form to receive direct download links for the thermal dataset release archive.

---

## 2. Directory Structure Setup

After downloading and extracting the dataset archive, arrange the directory structure locally as follows:

```text
datasets/
└── FLIR_ADAS/
    ├── coco.json          <-- Main COCO annotation file
    └── data/              <-- Directory containing all thermal images (.jpg / .jpeg / .png)
        ├── FLIR_00001.jpg
        ├── FLIR_00002.jpg
        └── ...
```

> **Note:** If your extracted archive names the annotation file `thermal_annotations.json` or `instances_val2017.json`, simply rename or copy it to `coco.json` inside `./datasets/FLIR_ADAS/`.

---

## 3. Running Data Preparation & Training

Once the folder structure matches the above layout, execute the data preparation and fine-tuning script:

```bash
python flir_real_thermal_yolov8_fine_tuning.py \
    --flir-root ./datasets/FLIR_ADAS \
    --output-dir ./output/flir_5cls_yolov8_finetuned \
    --epochs 150
```

The script will automatically perform:
- COCO JSON parsing into YOLO bounding box format (`.txt`).
- Category mapping (`person`, `car`, `motorcycle`, `train`, `truck`).
- Train/Validation split (85% train / 15% val).
- Rare class oversampling for `train`, `motorcycle`, and `truck`.
- Creation of `flir_real_thermal_5cls.yaml`.