"""
Convert COCO instance annotations (as output by the modified Labelme at https://github.com/mad4octos/LabelMe) 
to SAM2 frame masks in the MOSE/DAVIS dataset format.

DAVIS format [1]:

        - The annotations in each frame are stored in png format.
        - This png is stored indexed i.e. it has a single channel and each pixel has a value from 0 to 254 that 
          corresponds to a color palette attached to the png file.
        - When decoding the png i.e. the output of decoding should be a single channel image and it should not be 
          necessary to do any remap from RGB to indexes. 
        - The latter is crucial to preseve the index of each object so it can match to the correct object in evaluation.
        - Each pixel that belongs to the same object has the same value in this png map through the whole video.
        - Start at 1 for the first object, then 2, 3, 4 etc.
        - The background (not an object) has value 0.
        - Invalid/void pixels are stored with a 254 value.
    
Output structure [2]:
    output_dir/
    └── Annotations/
        ├── {video_name}/
        │   ├── 00001.png
        │   ├── 00002.png
        │   └── ...
        └── {video_name_2}/
            └── ...

Sources: 
 - [1] https://github.com/JonathonLuiten/TrackEval/blob/master/docs/DAVIS-format.txt    
 - [2] https://mose.video/#dataset

Note:
If multiple annotations share a frame and their masks overlap, the annotation with the higher ObjID wins (last-write).

Usage:
python convert_coco_to_davis_masks.py \
    --coco-file path/to/instances_train.json \
    --output-dir path/to/output \
    --video-name my_video

Arguments:
    --coco-file     Path to the COCO annotations file (instances_train.json).
    --output-dir    Root output directory. Masks are saved under <output-dir>/Annotations/<video-name>/.
    --video-name    Video/sequence name used as the subfolder under Annotations/.
"""

# Standard Library imports
import argparse
import json
from collections import defaultdict
from pathlib import Path

# External imports
import numpy as np
import numpy.typing as npt

# Local imports
from coco_types import CocoAnnotation, CocoImage, CompressedRLE
from pascal_colormap import pascal_colormap
from PIL import Image
from supervision.dataset.utils import rle_to_mask
from supervision.detection.utils.converters import polygon_to_mask


def _decode_segmentation(
    annotation: CocoAnnotation, image: CocoImage
) -> npt.NDArray[np.bool_]:
    """Return a boolean (H, W) mask for one annotation."""
    seg = annotation.segmentation
    resolution_wh = (image.width, image.height)

    if isinstance(seg, CompressedRLE):
        rle = np.array(seg.counts, dtype=np.int64)
        return rle_to_mask(rle=rle, resolution_wh=resolution_wh).astype(bool)

    # Polygon list (may contain multiple polygons for one annotation)
    combined = np.zeros((image.height, image.width), dtype=np.bool_)
    if not seg:
        return combined
    polygons = seg if isinstance(seg[0], list) else [seg]
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        combined |= polygon_to_mask(polygon=pts, resolution_wh=resolution_wh).astype(
            bool
        )
    return combined


def build_frame_mask(
    annotations: list[CocoAnnotation], image: CocoImage, id_offset: int = 0
) -> npt.NDArray[np.uint8]:
    """Combine multiple annotations into one (H, W) uint8 label array."""
    canvas = np.zeros((image.height, image.width), dtype=np.uint8)
    # Sort by ObjID. Higher IDs will paint over lower ones on overlap
    for ann in sorted(annotations, key=lambda a: a.attributes["ObjID"]):
        obj_id = ann.attributes["ObjID"]
        assert isinstance(obj_id, int), "ObjID is not an int!"
        mask = _decode_segmentation(ann, image)
        canvas[mask] = obj_id + id_offset
    return canvas


def save_label_mask(mask: npt.NDArray[np.uint8], out_path: Path) -> None:
    """Save a uint8 mask as an indexed (palette-mode) PNG.

    Modified from:
    https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/baselines/baseline_utils.py#L288
    """

    # Prepare palette
    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
    
    # Convert mask and save to a palette-indexed PNG
    img = Image.fromarray(mask).convert("P")
    img.putpalette(colmap.tolist())
    img.save(out_path)


def get_obj_ids(annotations: list[CocoAnnotation]) -> list[int]:
    """Return sorted list of ObjIDs present in annotations."""
    return sorted(
        ann.attributes["ObjID"] for ann in annotations if "ObjID" in ann.attributes
    )


def index_images_by_id(images: list[dict]) -> dict[int, CocoImage]:
    return {img["id"]: CocoImage.from_dict(img) for img in images}


def group_annotations_by_image_id(
    annotations: list[CocoAnnotation],
) -> dict[int, list[CocoAnnotation]]:
    """Group annotations by image_id, skipping any that lack or misuse an ObjID attribute."""
    image_id_to_ann: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for ann in annotations:
        if "ObjID" not in ann.attributes:
            print(f"Warning: annotation {ann.id} missing ObjID attribute — skipped")
            continue
        image_id_to_ann[ann.image_id].append(ann)
    return image_id_to_ann


def convert(coco_file: Path, output_dir: Path, video_name: str) -> None:

    out_dir = output_dir / "Annotations" / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_file, encoding="utf-8") as f:
        data = json.load(f)

    annotations: list[CocoAnnotation] = [
        CocoAnnotation.from_dict(a) for a in data["annotations"]
    ]
    image_id_to_ann = group_annotations_by_image_id(annotations)
    image_id_to_image = index_images_by_id(data["images"])
    all_obj_ids = get_obj_ids(annotations)

    if not all_obj_ids:
        raise ValueError("No annotations with ObjID found.")

    # In DAVIS, the background has value 0 and objects start at 1
    id_offset = 1 if 0 in all_obj_ids else 0

    if (max_obj_id := max(all_obj_ids) + id_offset) > 254:
        raise ValueError(
            f"ObjID {max_obj_id} conflicts with DAVIS void (254) or exceeds uint8 range."
        )

    print(f"ObjIDs found: {sorted(x + id_offset for x in set(all_obj_ids))}")
    print(f"Writing masks to: {out_dir}")
    print(f"Frames with annotations: {len(image_id_to_ann)}")

    for image_id, image_annotations in sorted(image_id_to_ann.items()):
        image_info = image_id_to_image[image_id]
        frame_mask = build_frame_mask(image_annotations, image_info, id_offset)
        # Image names need to be like "00000000.png" (not neccesarily with that number of zeros)
        # https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/training/dataset/vos_segment_loader.py#L116
        out_mask_path = out_dir / f"{image_info.filepath.stem}.png"
        save_label_mask(frame_mask, out_mask_path)

    print(f"Done — {len(image_id_to_ann)} mask(s) saved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Labelme COCO annotations to SAM2 palette PNG masks."
    )
    parser.add_argument(
        "--coco-file",
        type=Path,
        required=True,
        help="Path to the COCO annotation file (e.g. instances_train.json or instances_val.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory (masks go into <output-dir>/Annotations/<video-name>/)",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        required=True,
        help="Video/sequence name used as subfolder. Defaults to the coco-file's grandparent directory name.",
    )

    args = parser.parse_args()

    if not args.coco_file.exists():
        parser.error(f"COCO file not found: {args.coco_file}")

    convert(
        coco_file=args.coco_file, output_dir=args.output_dir, video_name=args.video_name
    )


if __name__ == "__main__":
    main()
