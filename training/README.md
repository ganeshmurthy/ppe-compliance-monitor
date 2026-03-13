# Training a Custom YOLO Model

This folder contains everything you need to train a YOLO model for badge detection (or other object classes) using your own images. The example dataset and notebook let you try training immediately.

## Overview

The PPE Compliance Monitor can use a custom-trained YOLO model for object detection. This training workflow helps you:

1. Prepare images and labels in YOLO format
2. Build a dataset and train a model
3. Use the trained model with the PPE app

## Folder Structure

```
training/
├── example/
│   ├── yolo_training.ipynb    # Run this notebook
│   ├── upload/                # Pre-populated example data (or your own)
│   │   ├── train_images/      # Training images
│   │   ├── train_labels/      # YOLO-format .txt labels
│   │   ├── val_images/        # Validation images
│   │   └── val_labels/        # Validation labels
│   └── data/                  # predefined_classes.txt for labeling tools
└── README.md
```

## Running the Notebook Locally

1. **Install JupyterLab:**
   ```bash
   pip install jupyterlab
   ```

2. **Navigate to the example folder and start Jupyter:**
   ```bash
   cd training/example
   jupyter lab
   ```

3. **Open `yolo_training.ipynb`** in the browser and run the cells in order.

The example dataset is already in `upload/`, so you can run the notebook immediately without copying files.

## Running on OpenShift Workbench

1. Open your Jupyter/Workbench workspace in OpenShift.
2. Upload or clone this repo into the workspace.
3. Navigate to `training/example/` and open `yolo_training.ipynb`.
4. Run the cells in order. Upload your own files to the `upload/` folders via the file browser if needed.

## Notebook Steps

| Step | Description |
|------|-------------|
| **1. Configuration** | Installs ultralytics, prompts for class names and output directory, creates `upload/` folders. |
| **2. Copy** | Copies images and labels from `upload/` into the YOLO dataset structure (`yolo_dataset/images/train`, etc.). |
| **3. Label Matching** | Ensures every image has a matching `.txt` label file; creates empty labels for images without annotations (negative examples). |
| **4. Generate YAML** | Creates `data.yaml` with paths and class names for Ultralytics. |
| **5. Train** | Trains a YOLOv8 model. Best checkpoint saved to `runs/detect/badge-demo/weights/best.pt`. |

## Dataset Requirements

### Label Format (YOLO)

Each `.txt` file has one line per object:

```
class_id  x_center  y_center  width  height
```

All coordinates are normalized (0–1). Images with no objects use an **empty** `.txt` file.

### Train vs. Validation

- **Train:** Images used to update model weights. Include both positive (with badges) and negative (without badges) examples.
- **Validation:** Held-out images for evaluation. Must be **different** from train images. Include both positive and negative examples so metrics (mAP, precision, recall) can be computed.

### Labeling Tools

- [Label Studio](https://labelstud.io) – self-hosted or cloud
- [Makesense.ai](https://www.makesense.ai) – browser-based
- [LabelImg](https://github.com/HumanSignal/labelImg) – desktop app

Export in **YOLO** format and place `.txt` files in `upload/train_labels/` and `upload/val_labels/`.

The `data/predefined_classes.txt` file lists class names for use with these tools.

## Using Your Own Data

1. Replace or add images in `upload/train_images/` and `upload/val_images/`.
2. Create matching `.txt` label files in `upload/train_labels/` and `upload/val_labels/`.
3. Run the notebook. Step 1 will prompt for class names if they differ from the default.

## Training Output

When training completes:

| Path | Description |
|------|-------------|
| `runs/detect/badge-demo/weights/best.pt` | Best model by validation metrics. Use for inference. |
| `runs/detect/badge-demo/weights/last.pt` | Final epoch checkpoint. |
| `runs/detect/badge-demo/results.png` | Loss and metrics plots. |

## Environment Variables (Optional)

Set these to skip prompts when running the notebook:

- `CLASSES` – Comma-separated class names (default: `Badge`)
- `OUTPUT_ROOT` – Output directory for YOLO dataset (default: `./yolo_dataset`)
