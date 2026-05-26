"""
Verify the expected dataset/ directory structure.

See README.md for the expected directory layout.
"""

# Standard Library imports
import argparse
import sys
from pathlib import Path

# Local imports
from check_utils import (
    BOLD,
    Reporter,
    _color,
    check_dir,
    check_files_matching,
    check_root,
)


def _obs_ids(parent: Path) -> set[str]:
    """Return names of immediate subdirectories of *parent* (empty set if missing)."""
    if not parent.is_dir():
        return set()
    return {d.name for d in parent.iterdir() if d.is_dir()}


def check_train(train_dir: Path, reporter: Reporter) -> None:
    """Validate the train split structure.

    Checks for the following subdirectories, along with their required files:
      - Annotations/<obs_id>/   *.png  (at least one)
      - JPEGImages/<obs_id>/    *.jpg  (at least one)

    Cross-checks that the set of observation IDs is identical in both trees.
    """
    annotations_root = train_dir / "Annotations"
    jpegs_root = train_dir / "JPEGImages"

    ann_ok = check_dir(annotations_root, reporter)
    jpg_ok = check_dir(jpegs_root, reporter)

    ann_ids = _obs_ids(annotations_root)
    jpg_ids = _obs_ids(jpegs_root)

    if not ann_ids and ann_ok:
        reporter.warn(
            f"Annotations/ has no observation subdirectories: {annotations_root}"
        )
    if not jpg_ids and jpg_ok:
        reporter.warn(f"JPEGImages/ has no observation subdirectories: {jpegs_root}")

    only_ann = ann_ids - jpg_ids
    only_jpg = jpg_ids - ann_ids
    if only_ann:
        reporter.error(
            f"Observations in Annotations/ but not in JPEGImages/: {sorted(only_ann)}"
        )
    if only_jpg:
        reporter.error(
            f"Observations in JPEGImages/ but not in Annotations/: {sorted(only_jpg)}"
        )

    for obs_id in sorted(ann_ids):
        print(f"\n  Observation (train): {_color(obs_id, BOLD)}")
        ann_obs = annotations_root / obs_id
        jpg_obs = jpegs_root / obs_id
        if ann_obs.is_dir():
            check_files_matching(ann_obs, "*.png", reporter)
        if jpg_obs.is_dir():
            check_files_matching(jpg_obs, "*.jpg", reporter)


def check_val(val_dir: Path, reporter: Reporter) -> None:
    """Validate the val split structure.

    Checks for the following subdirectories, along with their required files:
      - gt/Annotations/<obs_id>/                      *.png  (at least one)
      - trackers/<tracker_name>/Annotations/<obs_id>/ *.png  (at least one)

    Cross-checks that every gt observation ID appears in each tracker.
    """
    gt_annotations_root = val_dir / "gt" / "Annotations"
    trackers_root = val_dir / "trackers"

    gt_ok = check_dir(gt_annotations_root, reporter)
    check_dir(trackers_root, reporter)

    gt_ids = _obs_ids(gt_annotations_root)

    if not gt_ids and gt_ok:
        reporter.warn(
            f"gt/Annotations/ has no observation subdirectories: {gt_annotations_root}"
        )

    for obs_id in sorted(gt_ids):
        print(f"\n  Observation (val/gt): {_color(obs_id, BOLD)}")
        obs_dir = gt_annotations_root / obs_id
        if obs_dir.is_dir():
            check_files_matching(obs_dir, "*.png", reporter)

    tracker_dirs = (
        sorted(d for d in trackers_root.iterdir() if d.is_dir())
        if trackers_root.is_dir()
        else []
    )
    if not tracker_dirs:
        reporter.warn(f"trackers/ has no tracker subdirectories: {trackers_root}")
        return

    for tracker_dir in tracker_dirs:
        check_val_tracker(tracker_dir, gt_ids, reporter)


def check_val_tracker(tracker_dir: Path, gt_ids: set[str], reporter: Reporter) -> None:
    """Validate a single tracker directory inside val/trackers/.

    Checks for:
      - Annotations/<obs_id>/  *.png  (at least one per observation)

    Errors if any gt observation ID is absent; warns on extra IDs not in gt.
    """
    tracker_name = tracker_dir.name
    print(f"\n  Tracker: {_color(tracker_name, BOLD)}")

    ann_root = tracker_dir / "Annotations"
    if not check_dir(ann_root, reporter):
        return

    tracker_ids = _obs_ids(ann_root)

    if not tracker_ids:
        reporter.warn(f"Annotations/ has no observation subdirectories: {ann_root}")
        return

    missing = gt_ids - tracker_ids
    if missing:
        reporter.error(
            f"Tracker '{tracker_name}' is missing gt observations: {sorted(missing)}"
        )

    extra = tracker_ids - gt_ids
    if extra:
        reporter.warn(
            f"Tracker '{tracker_name}' has observations not in gt: {sorted(extra)}"
        )

    for obs_id in sorted(tracker_ids):
        obs_dir = ann_root / obs_id
        print(f"    Observation: {_color(obs_id, BOLD)}")
        if obs_dir.is_dir():
            check_files_matching(obs_dir, "*.png", reporter)


def check_dataset(root: Path, reporter: Reporter, split: str | None = None) -> None:
    splits = {split} if split else None
    check_root(root, "dataset", reporter, check_train, check_val, splits=splits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the dataset/ directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the dataset/ root directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Check only this split (default: check both train and val).",
    )
    args = parser.parse_args()

    reporter = Reporter()
    check_dataset(args.root.resolve(), reporter, split=args.split)
    reporter.summary()

    sys.exit(1 if reporter.errors else 0)


if __name__ == "__main__":
    main()
