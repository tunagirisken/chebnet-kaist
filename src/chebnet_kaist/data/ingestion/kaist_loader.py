"""KAIST Car-Hacking dataset loader."""

import re
from pathlib import Path

import pandas as pd

SUBDIR = "Car-Hacking Dataset"
NORMAL_TXT = "normal_run_data/normal_run_data.txt"

ATTACK_FILES: dict[str, str] = {
    "dos": "DoS_dataset.csv",
    "fuzzy": "Fuzzy_dataset.csv",
    "rpm": "RPM_dataset.csv",
    "gear": "gear_dataset.csv",
}

LABEL_MAP: dict[str, int] = {
    "normal": 0,
    "dos": 1,
    "fuzzy": 2,
    "rpm": 3,
    "gear": 4,
}

_NORMAL_LINE = re.compile(
    r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)\s+\d+\s+DLC:\s+(\d+)\s+(.*)"
)


def _read_attack_csv(path: Path) -> pd.DataFrame:
    """Read an attack CSV file and normalize column types.

    Args:
        path: Path to a KAIST attack CSV file.

    Returns:
        DataFrame with numeric timestamp, can_id, dlc, and binary flag.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "timestamp",
            "can_id",
            "dlc",
            "d0",
            "d1",
            "d2",
            "d3",
            "d4",
            "d5",
            "d6",
            "d7",
            "flag",
        ],
        dtype=str,
        low_memory=False,
    )
    df = df[df["flag"].isin(["R", "T"])].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["can_id"] = df["can_id"].apply(lambda value: int(str(value).strip(), 16))
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(0).astype(int)
    df["flag"] = (df["flag"] == "T").astype(int)
    df.dropna(subset=["timestamp", "can_id"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _read_normal_txt(path: Path) -> pd.DataFrame:
    """Read the normal_run_data.txt file in KAIST format.

    Args:
        path: Path to the normal traffic text file.

    Returns:
        DataFrame of normal CAN frames with label 0.
    """
    rows: list[list[float | int]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = _NORMAL_LINE.match(line)
            if match:
                rows.append([float(match.group(1)), int(match.group(2), 16), int(match.group(3))])

    df = pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc"])
    df["flag"] = 0
    df["label"] = 0
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_kaist(data_dir: str | Path, attack_type: str) -> pd.DataFrame:
    """Load a single KAIST class file and attach a graph-level label.

    Each row in the returned DataFrame is one CAN frame.

    Args:
        data_dir: Root directory containing the KAIST dataset.
        attack_type: One of ``normal``, ``dos``, ``fuzzy``, ``rpm``, ``gear``.

    Returns:
        CAN frame DataFrame for the requested class.

    Raises:
        FileNotFoundError: If the requested file does not exist.
        KeyError: If ``attack_type`` is not a known class name.
    """
    root = Path(data_dir) / SUBDIR

    if attack_type == "normal":
        path = root / NORMAL_TXT
        if not path.exists():
            raise FileNotFoundError(
                f"Normal data file not found: {path}. "
                "Run: python -m chebnet_kaist.cli.setup_data --instructions"
            )
        return _read_normal_txt(path)

    if attack_type not in ATTACK_FILES:
        raise KeyError(f"Unknown attack type: {attack_type}")

    path = root / ATTACK_FILES[attack_type]
    if not path.exists():
        raise FileNotFoundError(
            f"Attack file not found: {path}. "
            "The KAIST dataset is not included in git. Run:\n"
            "  python -m chebnet_kaist.cli.setup_data --instructions\n"
            "  python -m chebnet_kaist.cli.setup_data --verify"
        )

    df = _read_attack_csv(path)
    df["label"] = LABEL_MAP[attack_type]
    return df
