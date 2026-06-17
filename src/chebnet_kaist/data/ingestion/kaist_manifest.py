"""Expected KAIST Car-Hacking dataset layout and verification."""

from dataclasses import dataclass
from pathlib import Path

SUBDIR = "Car-Hacking Dataset"

# Official dataset page (manual download — no stable direct HTTP URLs).
OFFICIAL_URL = "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset"

ATTACK_CSV_FILES: tuple[str, ...] = (
    "DoS_dataset.csv",
    "Fuzzy_dataset.csv",
    "RPM_dataset.csv",
    "gear_dataset.csv",
)

NORMAL_TXT = "normal_run_data/normal_run_data.txt"
NORMAL_ARCHIVE = "normal_run_data.7z"


@dataclass(frozen=True)
class FileCheck:
    """Result of checking one expected dataset file."""

    relative_path: str
    exists: bool
    size_bytes: int


def expected_files(data_dir: Path) -> list[Path]:
    """Return all required file paths relative to ``data_dir``.

    Args:
        data_dir: Root KAIST data directory (e.g. ``data/kaist``).

    Returns:
        List of required absolute paths.
    """
    root = data_dir / SUBDIR
    paths = [root / name for name in ATTACK_CSV_FILES]
    paths.append(root / NORMAL_TXT)
    return paths


def verify_kaist_dataset(data_dir: Path) -> tuple[list[FileCheck], bool]:
    """Check whether all required KAIST files are present.

    Args:
        data_dir: Root KAIST data directory.

    Returns:
        Tuple of (per-file checks, all_present).
    """
    checks: list[FileCheck] = []
    for path in expected_files(data_dir):
        rel = str(path.relative_to(data_dir))
        if path.is_file():
            checks.append(FileCheck(rel, True, path.stat().st_size))
        else:
            checks.append(FileCheck(rel, False, 0))
    return checks, all(item.exists for item in checks)


def missing_files(data_dir: Path) -> list[str]:
    """Return relative paths of missing required files.

    Args:
        data_dir: Root KAIST data directory.

    Returns:
        List of missing relative paths.
    """
    checks, _ = verify_kaist_dataset(data_dir)
    return [item.relative_path for item in checks if not item.exists]


def total_size_mb(data_dir: Path) -> float:
    """Sum size of present required files in megabytes.

    Args:
        data_dir: Root KAIST data directory.

    Returns:
        Total size in MB for files that exist.
    """
    checks, _ = verify_kaist_dataset(data_dir)
    total = sum(item.size_bytes for item in checks if item.exists)
    return total / (1024 * 1024)
