"""
Run TrackEval over a val dataset prepared by prepare_val_dataset.py.

Expects the layout produced by prepare_val_dataset.py:

  <val_out_dir>/
  ├── gt/
  │   └── Annotations/
  │       └── <obs_id>/*.png
  └── trackers/
      └── <tracker_name>/
          └── Annotations/
              └── <obs_id>/*.png

Tracker names are discovered automatically from <val_out_dir>/trackers/.
TrackEval metric outputs are written to <results_dir> (default: results/ next to dataset/).

Usage
-----
    python validate.py \\
        --val-dir /path/to/dataset/val \\
        --trackeval-dir /path/to/TrackEval \\
        [--results-dir /path/to/results]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TrackEval over a prepared val dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--val-dir",
        required=True,
        help="Val dataset directory produced by prepare_val_dataset.py.",
    )
    p.add_argument(
        "--trackeval-dir",
        required=True,
        help="Root of the cloned TrackEval repo (contains scripts/run_davis.py).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["HOTA"],
        metavar="METRIC",
        help="TrackEval metrics to compute (default: HOTA).",
    )
    p.add_argument(
        "--num-cores",
        type=int,
        default=4,
        help="Parallel cores for TrackEval (default: 4).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Where to write TrackEval metric outputs. "
            "Defaults to a results/ directory next to dataset/."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    val_out_dir = Path(args.val_dir)
    trackers_dir = val_out_dir / "trackers"

    tracker_names = sorted(d.name for d in trackers_dir.iterdir() if d.is_dir())
    if not tracker_names:
        sys.exit(f"Error: no tracker directories found under {trackers_dir}")

    results_dir = (
        Path(args.results_dir) if args.results_dir else val_out_dir.parent.parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    trackeval_dir = Path(args.trackeval_dir)

    print(f"\n{'=' * 60}")
    print("Running TrackEval")
    print(f"{'=' * 60}\n")

    run(
        [
            sys.executable,
            str(trackeval_dir / "scripts" / "run_davis.py"),
            "--GT_FOLDER",
            str(val_out_dir / "gt" / "Annotations"),
            "--TRACKERS_FOLDER",
            str(trackers_dir),
            "--TRACKER_SUB_FOLDER",
            "Annotations",
            "--TRACKERS_TO_EVAL",
            *tracker_names,
            "--SPLIT_TO_EVAL",
            "val",
            "--METRICS",
            *args.metrics,
            "--USE_PARALLEL",
            "True",
            "--NUM_PARALLEL_CORES",
            str(args.num_cores),
            "--OUTPUT_FOLDER",
            str(results_dir),
        ]
    )

    print(
        f"\nDone. Results saved to: {results_dir} "
        f"({len(tracker_names)} tracker(s): {', '.join(tracker_names)})"
    )


if __name__ == "__main__":
    main()
