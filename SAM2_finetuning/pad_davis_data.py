"""
Pad GT and prediction DAVIS folders so they contain the same set of frames.

Frames present in one folder but missing from the other are filled with blank
(all-zero) palette-indexed PNGs. TrackEval errors if the two folders have a
different frame count.

Run this after `coco_to_sam2_masks.py` on both GT and predictions, and before
running TrackEval's `run_davis.py`.

Exactly 2 of the 3 arguments must be given:
  --gt-dir + --pred-dir    : pad each folder to match the other
  --gt-dir + --images-dir  : pad GT folder to match every frame in images dir
  --pred-dir + --images-dir: pad pred folder to match every frame in images dir

Usage:
    python scripts/pad_davis_predictions.py \
        --gt-dir   path/to/gt/Annotations/seq_name \
        --pred-dir path/to/trackers/my_tracker/Annotations/seq_name

    python scripts/pad_davis_predictions.py \
        --pred-dir path/to/trackers/my_tracker/Annotations/seq_name \
        --images-dir path/to/raw/frames/seq_name
"""

# Standard Library imports
import argparse
import sys
from pathlib import Path

# External imports
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from pascal_colormap import pascal_colormap

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def save_blank_mask(path: Path, height: int, width: int) -> None:
    blank = np.zeros((height, width), dtype=np.uint8)
    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
    img = Image.fromarray(blank).convert("P")
    img.putpalette(colmap.tolist())
    img.save(path)


def pad_missing(src: dict[str, Path], dst_dir: Path) -> int:
    """Add a blank mask to dst_dir for every frame present in src but not in dst_dir."""
    dst_names = {p.name for p in dst_dir.glob("*.png")}
    added = 0
    for name, src_path in src.items():
        if name not in dst_names:
            h, w = np.array(Image.open(src_path)).shape[:2]
            save_blank_mask(dst_dir / name, h, w)
            added += 1
    return added


def collect_pngs(directory: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(directory.glob("*.png"))}


def collect_images(directory: Path) -> dict[str, Path]:
    """Return {stem + '.png': path} for every image file in directory."""
    return {
        p.stem + ".png": p
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in _IMAGE_EXTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pad GT and/or prediction DAVIS folders to the same frame set."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        help="GT DAVIS folder (already converted PNGs)",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        help="Prediction DAVIS folder (already converted PNGs)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Raw images folder; used as the frame-name/dimension reference when one annotation dir is omitted",
    )
    args = parser.parse_args()

    given_args = sum(
        x is not None for x in (args.gt_dir, args.pred_dir, args.images_dir)
    )
    if given_args != 2:
        parser.error(
            "Exactly 2 of --gt-dir, --pred-dir, and --images-dir must be given."
        )

    # Pad each folder to match the other (used for tracking evaluation with TrackEval's run_davis.py)
    if args.gt_dir is not None and args.pred_dir is not None:
        gt_pngs = collect_pngs(args.gt_dir)
        pred_pngs = collect_pngs(args.pred_dir)
        added_to_pred = pad_missing(gt_pngs, args.pred_dir)
        added_to_gt = pad_missing(pred_pngs, args.gt_dir)
        if added_to_pred:
            print(f"Added {added_to_pred} blank frame(s) to {args.pred_dir}")
        if added_to_gt:
            print(f"Added {added_to_gt} blank frame(s) to {args.gt_dir}")
        if not added_to_pred and not added_to_gt:
            print("No padding needed — both folders already have the same frames.")

    # Pad gt or pred folder to match every frame in images dir (used for SAM2 finetuning)
    else:
        images = collect_images(args.images_dir)
        target_dir = args.gt_dir if args.gt_dir is not None else args.pred_dir
        added = pad_missing(images, target_dir)
        if added:
            print(f"Added {added} blank frame(s) to {target_dir}")
        else:
            print("No padding needed — folder already has all frames.")


if __name__ == "__main__":
    main()
