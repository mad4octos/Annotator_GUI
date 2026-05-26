"""
Prepare the train split of raw_data for SAM2 finetuning.

Expects this raw_data layout:

  raw_data/train/
  └── <observation_id>/
      ├── reviewed_annotations/
      │   ├── instances_train_v<N>.json   ← preferred GT (validated)
      │   └── instances_train.json        ← fallback GT
      └── images/
          └── train/
              └── *.jpg                   ← frame images

Steps performed for each observation:
  1. Convert GT COCO into DAVIS PNG masks.
  2. Copy JPEG frames to JPEGImages/<obs_id>/.
  3. Pad the Annotations folder so every frame has a corresponding mask.

Output layout (consumed by SAM2 finetuning):

  <train_out_dir>/
  ├── Annotations/
  │   └── <obs_id>/
  │       └── *.png
  └── JPEGImages/
      └── <obs_id>/
          └── *.jpg

Usage
-----
    python prepare_train_dataset.py \\
        --input-data-dir /path/to/raw_data/train \\
        --output-data-dir /path/to/dataset/train
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.resolve()


def run(cmd: list) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def discover_sequences(raw_data_dir: Path) -> list[dict]:
    """
    Walk raw_data_dir and return one dict per valid observation.

    An observation is valid when it has a GT COCO JSON (under reviewed_annotations/)
    and a non-empty images/train/ directory.

    Dict keys: obs_id, gt_coco, images_dir.
    """
    sequences = []
    for obs_dir in sorted(raw_data_dir.iterdir()):
        if not obs_dir.is_dir():
            continue

        obs_id = obs_dir.name

        # Pick highest-versioned instances_train_v<N>.json, fall back to instances_train.json
        gt_dir = obs_dir / "reviewed_annotations"
        versioned = sorted(
            gt_dir.glob("instances_train_v*.json"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
        )
        gt_coco = None
        if versioned:
            gt_coco = versioned[-1]
        elif (gt_dir / "instances_train.json").exists():
            gt_coco = gt_dir / "instances_train.json"

        if gt_coco is None:
            print(f"Warning: skipping {obs_id!r}; no GT COCO found under {gt_dir}")
            continue

        images_dir = obs_dir / "images" / "train"
        if not images_dir.is_dir() or not any(images_dir.glob("*.jpg")):
            print(
                f"Warning: skipping {obs_id!r}; no JPEG frames found under {images_dir}"
            )
            continue

        sequences.append(
            {"obs_id": obs_id, "gt_coco": gt_coco, "images_dir": images_dir}
        )

    return sequences


def copy_jpegs(images_dir: Path, dest_dir: Path) -> int:
    """Copy JPEG frames from images_dir into dest_dir. Returns the count of files copied."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(images_dir.glob("*.jpg")):
        dst = dest_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare train split of raw_data for SAM2 finetuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-data-dir",
        required=True,
        metavar="DIR",
        help=(
            "Train split of the raw_data directory (i.e. raw_data/train). Each subdirectory is "
            "treated as an observation and must contain reviewed_annotations/*.json "
            "and images/train/*.jpg."
        ),
    )
    p.add_argument(
        "--output-data-dir",
        required=True,
        metavar="DIR",
        help="Output directory where SAM2-ready Annotations and JPEGImages are written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_data_dir = Path(args.input_data_dir)
    sequences = discover_sequences(raw_data_dir)
    if not sequences:
        sys.exit(
            f"Error: no valid sequences found under {args.input_data_dir!r}. "
            "Each observation directory must contain reviewed_annotations/*.json "
            "and images/train/*.jpg."
        )

    train_out_dir = Path(args.output_data_dir)
    train_out_dir.mkdir(parents=True, exist_ok=True)

    n = len(sequences)

    for i, seq in enumerate(sequences):
        obs_id = seq["obs_id"]

        print(f"\n{'=' * 60}")
        print(f"Sequence {i + 1}/{n}: {obs_id}")
        print(f"{'=' * 60}")

        ann_out = train_out_dir / "Annotations"
        jpeg_out = train_out_dir / "JPEGImages" / obs_id

        # Step 1: GT COCO to DAVIS masks
        print("\n--- Step 1: Converting GT COCO to DAVIS ---")
        run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "convert_coco_to_davis_masks.py"),
                "--coco-file",
                str(seq["gt_coco"]),
                "--output-dir",
                str(train_out_dir),
                "--video-name",
                obs_id,
            ]
        )

        # Step 2: Copy JPEG frames
        print("\n--- Step 2: Copying JPEG frames ---")
        copied = copy_jpegs(seq["images_dir"], jpeg_out)
        if copied:
            print(f"Copied {copied} frame(s) to {jpeg_out}")
        else:
            print(f"No new frames to copy — {jpeg_out} already up to date.")

        # Step 3: Pad masks so every image frame has a corresponding .png
        print("\n--- Step 3: Padding missing mask frames ---")
        run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "pad_davis_data.py"),
                "--gt-dir",
                str(ann_out / obs_id),
                "--images-dir",
                str(seq["images_dir"]),
            ]
        )

    print(f"\nDone. Train dataset written to: {train_out_dir}")


if __name__ == "__main__":
    main()
