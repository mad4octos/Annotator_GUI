# Standard Library imports
import sys
from pathlib import Path
from typing import Callable

RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(_color(f"  [ERROR] {msg}", RED))

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(_color(f"  [WARN]  {msg}", YELLOW))

    def ok(self, msg: str) -> None:
        print(_color(f"  [OK]    {msg}", GREEN))

    def summary(self) -> None:
        print()
        print(BOLD + "=" * 60 + RESET if sys.stdout.isatty() else "=" * 60)
        e, w = len(self.errors), len(self.warnings)
        if e == 0 and w == 0:
            print(_color("All checks passed.", GREEN))
        else:
            if e:
                print(_color(f"{e} error(s) found.", RED))
            if w:
                print(_color(f"{w} warning(s) found.", YELLOW))
        print(BOLD + "=" * 60 + RESET if sys.stdout.isatty() else "=" * 60)


def check_dir(path: Path, reporter: Reporter, create: bool = False) -> bool:
    if path.is_dir():
        return True
    if create:
        path.mkdir(parents=True)
        reporter.warn(f"Created missing directory: {path}")
        return True
    reporter.error(f"Missing directory: {path}")
    return False


def check_file(path: Path, reporter: Reporter) -> bool:
    if path.is_file():
        return True
    reporter.error(f"Missing file: {path}")
    return False


def check_files_matching(directory: Path, pattern: str, reporter: Reporter) -> bool:
    if directory.is_dir() and any(directory.glob(pattern)):
        return True
    if directory.is_dir():
        reporter.error(f"No files matching '{pattern}' in: {directory}")
    return False


def check_root(
    root: Path,
    label: str,
    reporter: Reporter,
    check_train_fn: Callable[[Path, Reporter], None],
    check_val_fn: Callable[[Path, Reporter], None],
) -> None:
    """Check that root exists, then invoke per-split callbacks for train/ and val/.

    check_train_fn and check_val_fn receive the split directory and reporter.
    """
    print(_color(f"\nChecking {label} at: {root}", BOLD))

    if not root.is_dir():
        reporter.error(f"Root directory does not exist: {root}")
        return

    for split, fn in (("TRAIN", check_train_fn), ("VAL", check_val_fn)):
        split_dir = root / split.lower()
        print(f"\n{'─' * 40}")
        print(_color(split, BOLD))
        print(f"{'─' * 40}")
        if check_dir(split_dir, reporter):
            fn(split_dir, reporter)
