"""
Verify (and optionally create) the expected raw_data/ directory structure.

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
    check_file,
    check_files_matching,
    check_root,
)


def check_train_observation(obs_dir: Path, reporter: Reporter, create: bool) -> None:
    """Validate the expected file structure of a single train observation directory.

    Checks for:the following subdirectories, along with their required files:
      - manual_prompts/
      - predictions/
      - annotations/
      - reviewed_annotations/
      - images/train/

    Missing directories are created when `create` is True.
    """
    obs_id = obs_dir.name
    print(f"\n  Observation (train): {_color(obs_id, BOLD)}")

    # manual_prompts_dir = obs_dir / "manual_prompts"
    # if check_dir(manual_prompts_dir, reporter, create):
    #     check_file(manual_prompts_dir / f"{obs_id}_annotations.npy", reporter)

    # mpred = obs_dir / "predictions"
    # if check_dir(mpred, reporter, create):
    #     check_file(mpred / f"{obs_id}_masks.pkl", reporter)

    # annotations_dir = obs_dir / "annotations"
    # if check_dir(annotations_dir, reporter, create):
    #     check_file(annotations_dir / "instances_train.json", reporter)

    reviewed_annotations_dir = obs_dir / "reviewed_annotations"
    if check_dir(reviewed_annotations_dir, reporter, create):
        for fname in (
            # ".labelme_review.json",       # review metadata, not read by pipeline
            # "incorrect_predictions.json",  # review metadata, not read by pipeline
            "instances_train.json",
        ):
            check_file(reviewed_annotations_dir / fname, reporter)

        check_files_matching(
            reviewed_annotations_dir, "instances_train_v*.json", reporter
        )

    img_train = obs_dir / "images" / "train"
    if check_dir(img_train, reporter, create):
        check_files_matching(img_train, "*.jpg", reporter)


def check_val_observation(obs_dir: Path, reporter: Reporter, create: bool) -> None:
    """Validate the expected file structure of a single validation observation directory.

    Checks for the following subdirectories, along with their required files:
      - manual_prompts/
      - predictions/
      - annotations/
      - reviewed_annotations/
      - images/val/
      - trackers/<tracker_name>/

    Missing directories are created when `create` is True.
    """
    obs_id = obs_dir.name
    print(f"\n  Observation (val): {_color(obs_id, BOLD)}")

    # manual_prompts_dir = obs_dir / "manual_prompts"
    # if check_dir(manual_prompts_dir, reporter, create):
    #     check_file(manual_prompts_dir / f"{obs_id}_annotations.npy", reporter)

    # preds_dir = obs_dir / "predictions"
    # if check_dir(preds_dir, reporter, create):
    #     check_file(preds_dir / f"{obs_id}_masks.pkl", reporter)
    #     check_file(preds_dir / "run_info.json", reporter)

    # annotations_dir = obs_dir / "annotations"
    # if check_dir(annotations_dir, reporter, create):
    #     check_file(annotations_dir / "instances_val.json", reporter)

    reviewed_annotations_dir = obs_dir / "reviewed_annotations"
    if check_dir(reviewed_annotations_dir, reporter, create):
        for fname in (
            # ".labelme_review.json",       # review metadata, not read by pipeline
            # "incorrect_predictions.json",  # review metadata, not read by pipeline
            "instances_val.json",
        ):
            check_file(reviewed_annotations_dir / fname, reporter)
        check_files_matching(
            reviewed_annotations_dir, "instances_val_v*.json", reporter
        )

    # img_val = obs_dir / "images" / "val"  # optional — prepare_val_dataset.py sets images_path=None if absent
    # if check_dir(img_val, reporter, create):
    #     check_files_matching(img_val, "*.jpg", reporter)

    trackers_dir = obs_dir / "trackers"
    if not trackers_dir.is_dir():
        reporter.warn(f"No trackers/ directory yet in: {obs_dir}")
        if create:
            trackers_dir.mkdir(parents=True)
        return

    tracker_dirs = [d for d in trackers_dir.iterdir() if d.is_dir()]
    if not tracker_dirs:
        reporter.warn(f"trackers/ is empty in: {obs_dir}")
        return

    for tracker_dir in sorted(tracker_dirs):
        check_tracker_dir(tracker_dir, obs_id, reporter, create)


def check_tracker_dir(
    tracker_dir: Path, obs_id: str, reporter: Reporter, create: bool
) -> None:
    tracker_name = tracker_dir.name
    print(f"    Tracker: {_color(tracker_name, BOLD)}")

    # automatic_prompts/ (optional — may not exist if prompts come from elsewhere)
    # ap = tracker_dir / "automatic_prompts"
    # if ap.is_dir():
    #     check_file(ap / f"{obs_id}_annotations.json", reporter)
    #     check_file(ap / "run_info.json", reporter)
    # else:
    #     reporter.warn(f"Optional directory missing: {ap}")

    preds_dir = tracker_dir / "predictions"
    if check_dir(preds_dir, reporter, create):
        check_file(preds_dir / f"{obs_id}_masks.pkl", reporter)

    # annotations_dir = tracker_dir / "annotations"
    # if check_dir(annotations_dir, reporter, create):
    #     check_file(annotations_dir / "instances_val.json", reporter)  # generated by prepare_val_dataset.py


def _check_train_split(train_dir: Path, reporter: Reporter, create: bool) -> None:
    observation_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not observation_dirs:
        reporter.warn(f"train/ directory is empty: {train_dir}")
    for obs_dir in observation_dirs:
        check_train_observation(obs_dir, reporter, create)


def _check_val_split(val_dir: Path, reporter: Reporter, create: bool) -> None:
    observation_dirs = sorted(d for d in val_dir.iterdir() if d.is_dir())
    if not observation_dirs:
        reporter.warn(f"val/ directory is empty: {val_dir}")
    for obs_dir in observation_dirs:
        check_val_observation(obs_dir, reporter, create)


def check_raw_data(root: Path, reporter: Reporter, create: bool) -> None:
    check_root(
        root,
        "raw_data",
        reporter,
        check_train_fn=lambda d, r: _check_train_split(d, r, create),
        check_val_fn=lambda d, r: _check_val_split(d, r, create),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify (and optionally scaffold) the raw_data/ directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the raw_data/ root directory.",
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create missing directories (files are never auto-created).",
    )
    args = parser.parse_args()

    reporter = Reporter()
    check_raw_data(args.root.resolve(), reporter, create=args.create_missing)
    reporter.summary()

    sys.exit(1 if reporter.errors else 0)


if __name__ == "__main__":
    main()
