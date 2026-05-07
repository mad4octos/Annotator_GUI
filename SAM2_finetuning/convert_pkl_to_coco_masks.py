"""
Convert a SAM2 .pkl masks file to COCO JSON without requiring an annotations .npy file.

Usage:
    python convert_pkl_to_coco_masks.py --masks_path <path> --output_path <path> [--images_path <path>] [--obs_id <id>] [--class_name <name>] [--filename_num_zeros <n>] [--subset <name>]
"""

# Standard Library imports
import argparse
from collections import defaultdict
from pathlib import Path

# External imports
import pandas as pd

# Local imports
from convert_utils import (
    FrameIndex,
    MasksType,
    ObjectIndex,
    get_frame_chunks_df,
    load_categories,
    load_masks,
)
from dataset_builder import DatumaroDatasetBuilder


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

    builder = DatumaroDatasetBuilder(
        obs_id=obs_id,
        masks=masks,
        error_frames=[],
        chunked_df=chunked_df,
        annotations_df=annotations_df,
        label_categories=label_categories,
        export_root_path=output_path,
        images_path=images_path,
        classifier=None,
        blob_rules=[],
        window_size=5,
        anomaly_rules=[],
        correct_class=class_name,
        incorrect_class="",
        filename_num_zeros=filename_num_zeros,
        no_auto=True,
        subset=subset,
    )
    dataset = builder.build()

    dataset.export(str(output_path), format="coco_instances")
    print(f"COCO dataset written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAM2 .pkl masks to COCO JSON."
    )
    parser.add_argument("--masks_path", type=Path, required=True)
    parser.add_argument("--images_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--obs_id", type=str, default="obs")
    parser.add_argument("--class_name", type=str, default="fish")
    parser.add_argument("--filename_num_zeros", type=int, default=5)
    parser.add_argument("--subset", type=str, default="train")
    args = parser.parse_args()

    masks_pkl_to_coco(
        masks_path=args.masks_path,
        output_path=args.output_path,
        images_path=args.images_path,
        obs_id=args.obs_id,
        class_name=args.class_name,
        filename_num_zeros=args.filename_num_zeros,
        subset=args.subset,
    )
