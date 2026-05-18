"""
Prepare the val split of raw_data for TrackEval evaluation.

Expects this raw_data layout:

  raw_data/val/
  └── <observation_id>/
      ├── reviewed_annotations/
      │   ├── instances_val.json        # produced by AMC
      │   └── instances_val_v<N>.json   # produced by LabelMe (reviewed)
      ├── images/
      │   └── val/                      # Copied by AMC
      └── trackers/
          └── <tracker_name>/
              └── predictions/
                  └── *.pkl             # Produced by SAM2

Steps performed for each (observation, tracker) pair:
  1. Convert the SAM pickle to COCO via pkl_masks_to_coco.py.
  2. Convert GT COCO into DAVIS PNG masks.
  3. Convert predicted COCO into DAVIS PNG masks.
  4. Pad prediction folder so every GT frame has a corresponding mask.

Output layout (consumed by validate.py):

  <output_data_dir>/
  ├── gt/
  │   └── Annotations/
  │       └── <obs_id>/*.png
  └── trackers/
      └── <tracker_name>/
          └── Annotations/
              └── <obs_id>/*.png

Usage
-----
    python prepare_val_dataset.py \\
        --input-data-dir /path/to/raw_data/val \\
        --output-data-dir  /path/to/dataset/val
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.resolve()


def run(cmd: list) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def discover_sequences(raw_data_dir: Path) -> list[dict]:
    """
    Walk raw_data_dir and return one dict per valid (observation, tracker) pair.

    A pair is valid when the observation has a GT COCO JSON (under reviewed_annotations/)
    and the tracker subdirectory has at least one prediction .pkl (under predictions/).

    Dict keys: seq_name, tracker_name, gt_coco, pred_pkl, obs_id, images_path, tracker_dir.
    """
    sequences = []
    for obs_dir in sorted(raw_data_dir.iterdir()):
        if not obs_dir.is_dir():
            continue

        obs_id = obs_dir.name

        # GT COCO: pick highest-versioned instances_val_v<N>.json, fall back to instances_val.json
        gt_dir = obs_dir / "reviewed_annotations"
        versioned = sorted(
            gt_dir.glob("instances_val_v*.json"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
        )
        gt_coco = None
        if versioned:
            gt_coco = versioned[-1]
        elif (gt_dir / "instances_val.json").exists():
            gt_coco = gt_dir / "instances_val.json"

        if gt_coco is None:
            print(f"Warning: skipping {obs_id!r}; no GT COCO found under {gt_dir}")
            continue

        images_dir = obs_dir / "images" / "val"
        if not images_dir.exists():
            images_dir = None

        trackers_dir = obs_dir / "trackers"
        if not trackers_dir.exists():
            print(f"Warning: skipping {obs_id!r}; no trackers/ directory found")
            continue

        for tracker_dir in sorted(trackers_dir.iterdir()):
            if not tracker_dir.is_dir():
                continue

            tracker_name = tracker_dir.name
            pred_dir = tracker_dir / "predictions"
            pred_pkl = None
            if pred_dir.exists():
                pkl_files = sorted(pred_dir.glob("*.pkl"))
                if pkl_files:
                    pred_pkl = pkl_files[0]

            if pred_pkl is None:
                print(
                    f"Warning: skipping {obs_id!r}/{tracker_name!r}; "
                    f"no .pkl found under {pred_dir}"
                )
                continue

            sequences.append(
                {
                    "obs_id": obs_id,
                    "tracker_name": tracker_name,
                    "gt_coco": str(gt_coco),
                    "pred_pkl": str(pred_pkl),
                    "images_dir": images_dir,
                    "tracker_dir": tracker_dir,
                }
            )

    return sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare val split of raw_data for TrackEval evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-data-dir",
        required=True,
        metavar="DIR",
        help=(
            "Val split of the raw_data directory (i.e. raw_data/val). Each subdirectory is "
            "treated as an observation and must contain reviewed_annotations/*.json and "
            "trackers/<tracker_name>/predictions/*.pkl. Tracker names are discovered automatically."
        ),
    )
    p.add_argument(
        "--output-data-dir",
        required=True,
        help="Output directory where TrackEval-ready GT and tracker data are written.",
    )
    p.add_argument("--filename_num_zeros", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_data_dir = Path(args.input_data_dir)
    sequences = discover_sequences(raw_data_dir)
    if not sequences:
        sys.exit(
            f"Error: no valid sequences found under {args.input_data_dir!r}. "
            "Each observation directory must contain reviewed_annotations/*.json "
            "and trackers/<tracker_name>/predictions/*.pkl."
        )

    val_out_dir = Path(args.output_data_dir)
    gt_out = val_out_dir / "gt"
    val_out_dir.mkdir(parents=True, exist_ok=True)

    n = len(sequences)

    for i, seq in enumerate(sequences):
        obs_id = seq["obs_id"]
        tracker_name = seq["tracker_name"]
        tracker_dir_out = val_out_dir / "trackers" / tracker_name

        print(f"\n{'=' * 60}")
        print(f"Sequence {i + 1}/{n}: {obs_id}  [tracker: {tracker_name}]")
        print(f"{'=' * 60}")

        # Step 1: GT COCO to DAVIS
        print("\n--- Step 1: Converting GT COCO to DAVIS ---")
        run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "convert_coco_to_davis_masks.py"),
                "--coco-file",
                seq["gt_coco"],
                "--output-dir",
                str(gt_out),
                "--video-name",
                obs_id,
            ]
        )

        # Step 2: predictions PKL to COCO
        print("\n--- Step 2: Converting SAM2 prediction pickle files to COCO ---")
        tracker_dir = seq["tracker_dir"]
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "convert_pkl_to_coco_masks.py"),
            "--masks_path",
            seq["pred_pkl"],
            "--output_path",
            str(tracker_dir),
            "--obs_id",
            obs_id,
            "--subset",
            "val",
            "--filename_num_zeros",
            args.filename_num_zeros,
        ]
        if seq["images_dir"] is not None:
            cmd += ["--images_path", str(seq["images_dir"])]
        run(cmd)
        pred_coco = str(tracker_dir / "annotations" / "instances_val.json")

        # Step 3: Predictions COCO to DAVIS
        print("\n--- Step 3: Converting predictions COCO to DAVIS ---")
        run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "convert_coco_to_davis_masks.py"),
                "--coco-file",
                pred_coco,
                "--output-dir",
                str(tracker_dir_out),
                "--video-name",
                obs_id,
            ]
        )

        # Step 4: Pad missing GT/prediction frames
        print("\n--- Step 4: Padding missing GT/prediction frames ---")
        run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "pad_davis_data.py"),
                "--gt-dir",
                str(gt_out / "Annotations" / obs_id),
                "--pred-dir",
                str(tracker_dir_out / "Annotations" / obs_id),
            ]
        )

    print(f"\nDone. Val dataset written to: {val_out_dir}")


if __name__ == "__main__":
    main()
