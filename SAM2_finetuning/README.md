This folder contains scripts and configuration for finetuning SAM2 on fish segmentation data and evaluating the result with TrackEval.
The workflow has three stages: 
1. dataset preparation
   1.a manual preparation of the raw input data
   1.b automatic processing into the formats used for training and validation
2. SAM2 model finetuning
3. Validation of finetuned model

**Python version requirements:**
- Main scripts (`prepare_*.py`, `check_*.py`, `convert_*.py`, `validate.py`): **Python ≥ 3.9**
- TrackEval conda environment (Section 3.1): **Python 3.8** (isolated — do not use it to run the main scripts)

---

## Directory structures

### Input: `raw_data/`

Expected input layout.  Run `check_input_data_structure.py` to verify it (see Step 1 below).

```
raw_data/
├── train/
│   └── <observation_id>/
│       ├── manual_prompts/  # Produced by Annotator_GUI, consumed by SAM2 and AMC
│       │   └── <observation_id>_annotations.npy
│       ├── predictions/  # Produced by SAM2, consumed by AMC
│       │   └── <observation_id>_masks.pkl
│       ├── annotations/  # Produced by AMC, consumed by LabelMe
│       │   └── instances_train.json
│       ├── reviewed_annotations/  # Produced by LabelMe
│       │   ├── .labelme_review.json
│       │   ├── incorrect_predictions.json
│       │   ├── instances_train.json
│       │   └── instances_train_v*.json  (at least one)
│       └── images/  # Produced by AMC, consumed by LabelMe
│           └── train/
│               └── *.jpg
└── val/
    └── <observation_id>/
        ├── manual_prompts/
        │   └── <observation_id>_annotations.npy
        ├── predictions/
        │   ├── <observation_id>_masks.pkl
        │   └── run_info.json
        ├── annotations/
        │   └── instances_val.json
        ├── reviewed_annotations/
        │   ├── .labelme_review.json
        │   ├── incorrect_predictions.json
        │   ├── instances_val.json
        │   └── instances_val_v*.json  (at least one)
        ├── images/
        │   └── val/
        │       └── *.jpg
        └── trackers/
            └── <tracker_name>/
                ├── automatic_prompts/
                │   ├── <observation_id>_annotations.json
                │   └── run_info.json
                ├── predictions/
                │   └── <observation_id>_masks.pkl
                └── annotations/
                    └── instances_val.json
```

### Output: `dataset/` and `results/`

`dataset/` is consumed by SAM2 finetuning (train split) and TrackEval (val split).
SAM2 does not natively support validation; that is handled separately via TrackEval.
`results/` is written by TrackEval and kept separate from the dataset.

```
dataset/
├── train/
│   ├── Annotations/
│   │   ├── <observation_id_1>/
│   │   │   ├── 00000.png
│   │   │   └── ...
│   │   └── <observation_id_2>/
│   │       └── ...
│   └── JPEGImages/
│       ├── <observation_id_1>/
│       │   ├── 00000.jpg
│       │   └── ...
│       └── <observation_id_2>/
│           └── ...
└── val/
    ├── gt/
    │   └── Annotations/
    │       ├── <observation_id_a>/
    │       │   └── *.png
    │       └── <observation_id_b>/
    │           └── ...
    └── trackers/
        └── <tracker_name>/
            └── Annotations/
                ├── <observation_id_a>/
                │   └── *.png
                └── <observation_id_b>/
                    └── ...

results/                        # TrackEval metric outputs (written by validate.py)
```

---

## Prerequisites

### herbfishCV

Several scripts depend on modules from `herbfishCV` (`coco_types`, `convert_utils`, `dataset_builder`). Clone and install it once before running any of the steps below.

> **Note:** One of herbfishCV's dependencies requires Rust. Install it first if you don't have it:
> ```bash
> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
> . "$HOME/.cargo/env"
> ```

```bash
git clone https://github.com/mad4octos/herbfishCV
pip install -r herbfishCV/requirements.txt
```

Set `PYTHONPATH` at the start of each shell session before running the scripts:

```bash
export PYTHONPATH=/path/to/herbfishCV        # Linux / macOS
```
```powershell
$env:PYTHONPATH = "C:\path\to\herbfishCV"   # Windows (PowerShell)
```

---

## 1. Prepare the dataset

**Scripts used:**

- **`check_input_data_structure.py`** — Validates that `raw_data/` matches the expected input layout before conversion begins.

- **`check_output_data_structure.py`** — Validates that `dataset/` matches the expected output layout consumed by SAM2 finetuning and TrackEval.

- **`convert_coco_to_davis_masks.py`** — Converts a reviewed COCO JSON annotation file (from Labelme) into palette-indexed PNG masks in the DAVIS format.

- **`pad_davis_data.py`** — SAM2 requires one mask per frame, but annotations typically cover only a subset of frames. This script fills in all-zero (blank) palette-indexed PNGs for any frame that has no annotation.

- **`convert_pkl_to_coco_masks.py`** — Converts a SAM2 `.pkl` masks file to COCO JSON without requiring an annotations `.npy` file.

- **`prepare_train_dataset.py`** — Automates the full train-split preparation pipeline: converts GT COCO into DAVIS palette PNGs, copies JPEG frames to `JPEGImages/`, and pads the annotations folder so every frame has a corresponding mask.

- **`prepare_val_dataset.py`** — Automates the full val-split preparation pipeline: converts tracker `.pkl` predictions into COCO JSON, converts GT and predicted COCO into DAVIS palette PNGs, and pads both folders so every GT frame has a corresponding prediction mask.

### Step 1 — Prepare the raw data

Manually prepare according to section "Input: `raw_data/`"

### Step 2 — Check the input structure

Walk every observation directory under `train/` and `val/` and reports missing files or subdirectories.

```bash
python check_input_data_structure.py /path/to/raw_data
```

Add `--create-missing` to scaffold any missing directories (files are not auto-created).

### Step 3 — Prepare the train dataset

Convert reviewed annotations to DAVIS masks, copy JPEG frames, and pad any unannotated frames with blank masks.

```bash
python prepare_train_dataset.py \
    --input-data-dir /path/to/raw_data/train \
    --output-data-dir /path/to/dataset/train
```

### Step 4 — Prepare the val dataset

Convert tracker `.pkl` predictions to COCO JSON, GT and predicted COCO annotations to DAVIS masks, and pads both folders so every GT frame has a corresponding prediction mask.

Ensure `PYTHONPATH` includes `herbfishCV` (see [Prerequisites](#prerequisites)).

```bash
python prepare_val_dataset.py \
    --input-data-dir /path/to/raw_data/val \
    --output-data-dir /path/to/dataset/val
```

### Step 5 — Check the output structure

```bash
python check_output_data_structure.py /path/to/dataset
```

---

## 2. Finetune SAM2

### 2.1 Install SAM2 and download weights

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[dev]"
cd checkpoints && ./download_ckpts.sh
```

### 2.2 herbfishCV

`herbfishCV` must already be cloned, installed, and on your `PYTHONPATH` — see [Prerequisites](#prerequisites).

### 2.3 Edit the YAML configuration

In `configs/sam2.1_hiera_large_finetune.yaml`, update the two dataset path variables to match your `dataset/train/`:

```yaml
img_folder: /path/to/dataset/train/JPEGImages   # dataset/train/JPEGImages
gt_folder:  /path/to/dataset/train/Annotations  # dataset/train/Annotations
```

### 2.4 Move the YAML configuration into SAM2

```bash
mv configs/sam2.1_hiera_large_finetune.yaml sam2/configs/
```

### 2.5 Start training

```bash
cd sam2
python training/train.py \
    -c configs/sam2.1_hiera_large_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

---

## 3. Validate SAM2

By this point `prepare_val_dataset.py` has already been run in Step 4, so `dataset/val/` is ready.
`validate.py` runs TrackEval over that directory and prints results.

### 3.1 Prerequisites

TrackEval requires Python 3.8. Use conda to create an isolated environment for it:

```bash
conda create -n trackeval_env python=3.8 -y
conda activate trackeval_env

pip install matplotlib-inline ipython numpy==1.18.1 scipy==1.4.1 \
    pycocotools==2.0.7 matplotlib==3.2.1 opencv_python==4.4.0.46 \
    scikit_image==0.16.2 pytest==6.0.1 Pillow==8.1.2 tqdm==4.64.0 tabulate
```

Clone TrackEval:

```bash
git clone https://github.com/JonathonLuiten/TrackEval
```

`herbfishCV` must already be cloned, installed, and on your `PYTHONPATH` — see [Prerequisites](#prerequisites).

`raw_data/val/` must be fully populated (see the input structure above), with at least one tracker's
`predictions/*.pkl` present under each observation directory.

### 3.2 Run validation

```bash
python validate.py \
    --val-dir       /path/to/dataset/val \
    --trackeval-dir /path/to/TrackEval \
    [--results-dir  /path/to/results]     # optional; default: results/ next to dataset/
```

Results are printed to stdout and written to `results/` (or the path given via `--results-dir`).
