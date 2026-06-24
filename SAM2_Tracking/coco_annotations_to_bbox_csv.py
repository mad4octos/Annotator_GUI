"""
Convert a COCO annotation file to a SAM2 bounding-box prompt CSV.

The output CSV has the following columns:

    ObjID (index)  Frame  BBox                        ObjType
    -------------  -----  --------------------------  ----------
    1              1      [100.0, 150.0, 300.0, 400.0]  Parrotfish

- ObjID   : from annotation.attributes["ObjID"]
- Frame   : int(Path(image["file_name"]).stem)  — raw filename number
- BBox    : [x_min, y_min, x_max, y_max] stored as a string; converted from COCO [x, y, w, h]
- ObjType : category name looked up via annotation.category_id

Reading back for SAM2:

    import ast, pandas as pd
    df = pd.read_csv("output.csv", index_col="ObjID")
    df["BBox"] = df["BBox"].apply(ast.literal_eval)
    segmenter.add_box_annotations(annotations=df)

Usage:
    python coco_annotations_to_bbox_csv.py --coco-file path/to/instances.json
    python coco_annotations_to_bbox_csv.py --coco-file path/to/instances.json --output out.csv
"""

# Standard Library imports
import argparse
import json
from pathlib import Path

# External imports
import pandas as pd

# Local imports
# Needs herbfishCV to be on the PYTHONPATH
from coco_types import CocoAnnotation, CocoFile, CocoImage


def coco_to_dataframe(coco_path: Path) -> pd.DataFrame:
    """
    Parse a COCO JSON file and return a DataFrame of bounding-box annotations
    ready for SAM2 prompting.

    Parameters
    ----------
    coco_path : Path
        Path to the COCO JSON annotation file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ObjID with columns Frame, BBox, ObjType.
        BBox values are Python lists [x_min, y_min, x_max, y_max].

    Raises
    ------
    ValueError
        If no valid annotations with ObjID are found.
    """
    with open(coco_path, encoding="utf-8") as f:
        data: CocoFile = json.load(f)

    id_to_image: dict[int, CocoImage] = {
        img["id"]: CocoImage.from_dict(img) for img in data["images"]
    }
    id_to_category_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in data["categories"]
    }

    rows: list[dict] = []
    skipped = 0
    for ann_dict in data["annotations"]:
        ann = CocoAnnotation.from_dict(ann_dict)
        if "ObjID" not in ann.attributes:
            print(f"Warning: annotation {ann.id} missing ObjID attribute — skipped")
            skipped += 1
            continue
        obj_id = ann.attributes["ObjID"]
        image = id_to_image[ann.image_id]
        # -1 because the filenames are 1-indexed, but SAM2 expects them 0-indexed
        frame = int(image.filepath.stem) - 1

        x, y, w, h = ann.bbox
        bbox = [int(x), int(y), int(x + w), int(y + h)]

        rows.append(
            {
                "ObjID": obj_id,
                "Frame": frame,
                "BBox": bbox,
                "ObjType": id_to_category_name[ann.category_id],
            }
        )

    if not rows:
        raise ValueError("No valid annotations with ObjID found in the COCO file.")

    df = (
        pd.DataFrame(rows)
        .astype({"ObjID": int, "Frame": int, "ObjType": str})
        .sort_values(["ObjID", "Frame"])
        .set_index("ObjID")
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a COCO annotation file to a SAM2 bounding-box prompt CSV."
    )
    parser.add_argument(
        "--coco-file",
        type=Path,
        required=True,
        help="Path to the COCO annotation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the output CSV. "
            "Defaults to <coco_file_stem>_sam2_bbox.csv next to the input file."
        ),
    )
    args = parser.parse_args()

    if not args.coco_file.exists():
        parser.error(f"COCO file not found: {args.coco_file}")

    output_path: Path = args.output or (
        args.coco_file.parent / f"{args.coco_file.stem}_sam2_bbox.csv"
    )

    df = coco_to_dataframe(args.coco_file)

    # Store BBox as a string so it round-trips through CSV cleanly
    df_out = df.copy()
    df_out["BBox"] = df_out["BBox"].apply(str)
    df_out.to_csv(output_path)

    print(
        f"Wrote {len(df)} annotation(s) for {df.index.nunique()} unique ObjID(s) to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
