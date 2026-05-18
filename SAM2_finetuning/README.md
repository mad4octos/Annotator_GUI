This folder contains scripts and configuration for finetuning SAM2 on fish segmentation data and evaluating the result with TrackEval.
The workflow has three stages: 
1. Dataset preparation
   1.a Manual preparation of the raw input data
   1.b Automatic processing into the formats used for training and validation
2. SAM2 model finetuning
3. Validation of finetuned model(s)

**Python version requirements:**
- Main scripts (`prepare_*.py`, `check_*.py`, `convert_*.py`, `validate.py`): **Python ≥ 3.9**
- TrackEval conda environment (Section 3.1): **Python 3.8** (isolated — do not use it to run the main scripts)

---

## Directory structures

### Input: `raw_data/`

Expected input layout.  You will be able to run `check_input_data_structure.py` during Step 1 below to verify it.

Train and val split assignments must be decided before running the AMC, so that the `--subset` flag can be passed from the start and the correct suffixes (`_train` / `_val`) are applied to output files and dirs.

```
raw_data/
├── train/
│   └── <observation_id>/
│       ├── reviewed_annotations/  
│       │   ├── instances_train.json  # Produced by AMC
│       │   └── instances_train_v*.json  # Produced by LabelMe
│       └── images/  # Produced by AMC
│           └── train/
│               └── *.jpg
└── val/
    └── <observation_id>/
        ├── reviewed_annotations/
        │   ├── instances_val.json  # Produced by AMC
        │   └── instances_val_v*.json  # Produced by LabelMe
        ├── images/  # Produced by AMC
        │   └── val/
        │       └── *.jpg
        └── trackers/
            ├── <tracker_name>/
            │   └── predictions/
            │       └── <observation_id>_masks.pkl  # Produced by SAM2
            └── ...
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

### Project structure

Suggested project layout, where `<root>/` is the working directory (e.g. `/scratch/alpine/user`):
```
<root>/
├── raw_data/
│   └── ...
├── dataset/
│   └── ...
├── herbfishCV/
│   └── ...
├── Annotator_GUI/
│   ├── SAM2_finetuning/
│   │   └── ...
│   └── SAM2_Tracking/
│       └── ...
├── sam2/
│   └── ...
├── TrackEval/
│   └── ...
└── results/
    └── ...
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
cd <root>/
git clone https://github.com/mad4octos/herbfishCV
pip install -r herbfishCV/requirements.txt
```

Set `PYTHONPATH` at the start of each shell session before running the scripts:

```bash
export PYTHONPATH="<root>/herbfishCV:<root>/herbfishCV/scripts"
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
cd <root>/Annotator_GUI/SAM2_finetuning/
python check_input_data_structure.py <root>/raw_data
```

Add `--create-missing` to scaffold any missing directories (files are not auto-created).

### Step 3 — Prepare the train dataset

Convert reviewed annotations to DAVIS masks, copy JPEG frames, and pad any unannotated frames with blank masks.

```bash
cd <root>/Annotator_GUI/SAM2_finetuning/
python prepare_train_dataset.py \
    --input-data-dir <root>/raw_data/train \
    --output-data-dir <root>/dataset/train
```

### Step 4 — Prepare the val dataset

Convert tracker `.pkl` predictions to COCO JSON, GT and predicted COCO annotations to DAVIS masks, and pads both folders so every GT frame has a corresponding prediction mask.

Ensure `PYTHONPATH` includes `herbfishCV` and `herbfishCV/scripts` (see [Prerequisites](#prerequisites)).

```bash
cd <root>/Annotator_GUI/SAM2_finetuning/
python prepare_val_dataset.py \
    --input-data-dir <root>/raw_data/val \
    --output-data-dir <root>/dataset/val \
    [--filename_num_zeros N]   # zero-padding width for output filenames (default: 5)
```

### Step 5 — Check the output structure

```bash
cd <root>/Annotator_GUI/SAM2_finetuning/
python check_output_data_structure.py <root>/dataset
```

---

## 2. Finetune SAM2

### 2.1 Install SAM2 and download weights

It's assumed that you are in directory `<root>`

```bash
cd <root>/
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[dev]"
cd checkpoints && ./download_ckpts.sh
```

### 2.2 herbfishCV

`herbfishCV` must already be cloned, installed, and on your `PYTHONPATH` — see [Prerequisites](#prerequisites).

### 2.3 Edit the YAML configuration

In `<root>/Annotator_GUI/SAM2_finetuning/configs/sam2.1_hiera_large_finetune.yaml`, update the two dataset path variables to match your `dataset/train/`:

```yaml
img_folder: <root>/dataset/train/JPEGImages   # dataset/train/JPEGImages
gt_folder:  <root>/dataset/train/Annotations  # dataset/train/Annotations
```

### 2.4 Move the YAML configuration to the SAM2 repository

```bash
cp <root>/Annotator_GUI/SAM2_finetuning/configs/sam2.1_hiera_large_finetune.yaml <root>/sam2/sam2/configs/
```

### 2.5 Start training

```bash
cd <root>/sam2/
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
cd <root>/
git clone https://github.com/JonathonLuiten/TrackEval
```

`herbfishCV` must already be cloned, installed, and on your `PYTHONPATH` — see [Prerequisites](#prerequisites).

`raw_data/val/` must be fully populated (see the input structure above), with at least one tracker's
`trackers/<tracker_name>/predictions/*.pkl` present under each observation directory.

### 3.2 Run validation

```bash
cd <root>/Annotator_GUI/SAM2_finetuning/
python validate.py \
    --val-dir       <root>/dataset/val \
    --trackeval-dir <root>/TrackEval \
    [--results-dir  <root>/results]     # optional; default: results/ next to dataset/
```

Results are printed to stdout and written to `results/` (or the path given via `--results-dir`).
