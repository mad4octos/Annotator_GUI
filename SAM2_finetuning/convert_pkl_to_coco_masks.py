"""
Converts a SAM2 .pkl masks file directly to a COCO JSON dataset, without requiring an
annotations .npy file. Object IDs and frame ranges are derived from the masks themselves.
All masks are accepted as-is (no automatic cleaning), keeping the largest blob per object
per frame.

Usage:
    python convert_pkl_to_coco_masks.py \
        --masks_path /path/to/masks.pkl \
        --output_path /path/to/output \
        [--images_path /path/to/frames] \
        [--obs_id <id>] \
        [--class_name <name>] \
        [--filename_num_zeros <n>] \
        [--subset <name>]

Arguments:
    --masks_path          Path to the SAM2 masks .pkl file.
    --output_path         Root output directory. COCO JSON is written under <output_path>/coco/.
    --images_path         (Optional) Path to the directory containing the extracted image frames.
                          When omitted, image dimensions are inferred from the mask tensors and
                          no image files are read.
    --obs_id              Observation identifier used as the dataset name (default: obs).
    --class_name          Object class name (default: fish).
    --filename_num_zeros  Zero-padding width for frame filenames (default: 5).
    --subset              Dataset subset name, e.g. train or val (default: train).
"""

# Standard Library imports
import argparse
import json
from collections import defaultdict
from pathlib import Path

# External imports
import pandas as pd

# Local imports
from coco_types import CocoFile, CocoImage
from convert_utils import (
    FrameIndex,
    MasksType,
    ObjectIndex,
    get_frame_chunks_df,
    load_categories,
    load_masks,
)
from dataset_builder import DatumaroDatasetBuilder


def infer_frame_step_from_coco(coco_file: Path) -> int | None:
    """ """
    with open(coco_file) as f:
        coco: CocoFile = json.load(f)

    frame_numbers: list[int] = []
    for img in coco.get("images", []):
        coco_image = CocoImage.from_dict(img)
        frame_numbers.append(int(coco_image.filepath.stem))
    frame_numbers.sort()

    if len(frame_numbers) < 2:
        return None

    diffs = [
        frame_numbers[i + 1] - frame_numbers[i] for i in range(len(frame_numbers) - 1)
    ]
    assert len(set(diffs)) == 1, f"Inconsistent frame step in COCO file: {set(diffs)}"
    return diffs[0]


def build_annotations_df(masks: MasksType, class_name: str) -> pd.DataFrame:
    obj_frames: defaultdict[ObjectIndex, list[FrameIndex]] = defaultdict(list)
    for frame_idx, frame_masks in masks.items():
        for obj_id in frame_masks:
            obj_frames[obj_id].append(frame_idx)

    obj_first = {obj_id: min(frames) for obj_id, frames in obj_frames.items()}
    obj_last = {obj_id: max(frames) for obj_id, frames in obj_frames.items()}

    rows: list[dict] = []
    for obj_id in sorted(obj_first):
        rows += [
            {
                "ObjID": str(obj_id),
                "ObjType": class_name,
                "Frame": obj_first[obj_id],
                "ClickType": 3,
                "Location": [0.0, 0.0],
            },
            {
                "ObjID": str(obj_id),
                "ObjType": class_name,
                "Frame": obj_last[obj_id],
                "ClickType": 4,
                "Location": [0.0, 0.0],
            },
        ]
    return pd.DataFrame(rows).astype(
        {"ObjID": str, "ObjType": str, "ClickType": int, "Frame": int}
    )


def masks_pkl_to_coco(
    masks_path: Path,
    coco_file: Path,
    output_path: Path,
    images_path: Path | None = None,
    obs_id: str = "obs",
    class_name: str = "fish",
    filename_num_zeros: int = 5,
    subset: str = "train",
) -> None:
    masks = load_masks(masks_path)
    annotations_df = build_annotations_df(masks, class_name)
    label_categories = load_categories(annotations_df)
    chunked_df = get_frame_chunks_df(annotations_df)
    frame_step = infer_frame_step_from_coco(coco_file)

    builder = DatumaroDatasetBuilder(
        obs_id=obs_id,
        masks=masks,
        error_frames=[],
        chunked_df=chunked_df,
        annotations_df=annotations_df,
        label_categories=label_categories,
        export_root_path=output_path,
        images_path=images_path,
        filename_num_zeros=filename_num_zeros,
        no_auto=True,
        subset=subset,
    )
    builder.frame_step = frame_step
    dataset = builder.build()

    dataset.export(str(output_path), format="coco_instances")
    print(f"COCO dataset written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAM2 .pkl masks to COCO JSON."
    )
    parser.add_argument("--masks_path", type=Path, required=True)
    parser.add_argument("--coco-file", type=Path, required=True)
    parser.add_argument("--images_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--obs_id", type=str, default="obs")
    parser.add_argument("--class_name", type=str, default="fish")
    parser.add_argument("--filename_num_zeros", type=int, default=5)
    parser.add_argument("--subset", type=str, default="train")
    args = parser.parse_args()

    masks_pkl_to_coco(
        masks_path=args.masks_path,
        coco_file=args.coco_file,
        output_path=args.output_path,
        images_path=args.images_path,
        obs_id=args.obs_id,
        class_name=args.class_name,
        filename_num_zeros=args.filename_num_zeros,
        subset=args.subset,
    )
