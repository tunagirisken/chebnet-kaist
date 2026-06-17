"""CLI to verify and prepare external KAIST dataset files (not stored in git)."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from chebnet_kaist.data.ingestion.kaist_manifest import (
    ATTACK_CSV_FILES,
    NORMAL_ARCHIVE,
    NORMAL_TXT,
    OFFICIAL_URL,
    SUBDIR,
    missing_files,
    total_size_mb,
    verify_kaist_dataset,
)

_DEFAULT_DATA_DIR = Path("data/kaist")


def _print_instructions(data_dir: Path) -> None:
    """Print manual download instructions for the KAIST dataset."""
    target = data_dir / SUBDIR
    print("KAIST Car-Hacking dataset is NOT included in this git repository.")
    print("Files total ~900 MB and exceed GitHub size limits.\n")
    print("1. Open the official download page:")
    print(f"   {OFFICIAL_URL}\n")
    print("2. Download all attack CSV files and the normal-run archive.")
    print("3. Place files in this directory:")
    print(f"   {target.resolve()}\n")
    print("   Required layout:")
    for name in ATTACK_CSV_FILES:
        print(f"     {name}")
    print(f"     {NORMAL_TXT}")
    print(f"     (optional archive) {NORMAL_ARCHIVE}\n")
    print("4. Run verification:")
    print("   python -m chebnet_kaist.cli.setup_data --verify\n")
    print("5. If you only have the .7z archive for normal data:")
    print("   python -m chebnet_kaist.cli.setup_data --extract-normal")


def _extract_normal_archive(data_dir: Path) -> bool:
    """Extract normal_run_data.7z when the txt file is missing.

    Args:
        data_dir: Root KAIST data directory.

    Returns:
        True if normal_run_data.txt exists after this call.
    """
    root = data_dir / SUBDIR
    archive = root / NORMAL_ARCHIVE
    txt_path = root / NORMAL_TXT

    if txt_path.is_file():
        print(f"Already present: {txt_path}")
        return True

    if not archive.is_file():
        print(f"Archive not found: {archive}")
        print("Download normal_run_data.7z from the official page first.")
        return False

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    for cmd in (
        ["7z", "x", str(archive), f"-o{txt_path.parent}", "-y"],
        ["7za", "x", str(archive), f"-o{txt_path.parent}", "-y"],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        print(f"Extracting with {cmd[0]} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and txt_path.is_file():
            print(f"Extracted: {txt_path}")
            return True
        if result.stderr:
            print(result.stderr.strip())

    try:
        import py7zr  # type: ignore[import-untyped]
    except ImportError:
        print("Install py7zr for Python-based extraction:")
        print('  pip install -e ".[data]"')
        print("Or install p7zip and rerun with --extract-normal.")
        return False

    print("Extracting with py7zr ...")
    with py7zr.SevenZipFile(archive, mode="r") as archive_handle:
        archive_handle.extractall(path=txt_path.parent)

    if txt_path.is_file():
        print(f"Extracted: {txt_path}")
        return True

    print("Extraction finished but normal_run_data.txt was not found.")
    return False


def _copy_from_source(source: Path, data_dir: Path) -> int:
    """Copy dataset files from a local source directory.

    Args:
        source: Directory containing ``Car-Hacking Dataset/`` or the files directly.
        data_dir: Target KAIST root (``data/kaist``).

    Returns:
        Number of files copied.
    """
    if (source / SUBDIR).is_dir():
        source_root = source / SUBDIR
    else:
        source_root = source

    target_root = data_dir / SUBDIR
    target_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in ATTACK_CSV_FILES:
        src = source_root / name
        dst = target_root / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {name}")
            copied += 1

    for rel in (NORMAL_TXT, NORMAL_ARCHIVE):
        src = source_root / rel
        dst = target_root / rel
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied: {rel}")
            copied += 1

    return copied


def _run_verify(data_dir: Path) -> int:
    """Print verification report and return exit code.

    Args:
        data_dir: Root KAIST data directory.

    Returns:
        0 if complete, 1 if files are missing.
    """
    checks, complete = verify_kaist_dataset(data_dir)
    print(f"Dataset directory: {data_dir.resolve()}\n")

    for item in checks:
        if item.exists:
            size_mb = item.size_bytes / (1024 * 1024)
            print(f"  [OK]   {item.relative_path} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISS] {item.relative_path}")

    print()
    if complete:
        print(f"All required files present ({total_size_mb(data_dir):.0f} MB total).")
        return 0

    print(f"Missing {len(missing_files(data_dir))} file(s). Run with --instructions.")
    return 1


def main() -> None:
    """Parse CLI arguments and run dataset setup actions."""
    parser = argparse.ArgumentParser(
        description="Verify or prepare the external KAIST Car-Hacking dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="KAIST root directory (default: data/kaist)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that all required files exist",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download and layout instructions",
    )
    parser.add_argument(
        "--extract-normal",
        action="store_true",
        help="Extract normal_run_data.7z if txt is missing",
    )
    parser.add_argument(
        "--copy-from",
        type=Path,
        metavar="PATH",
        help="Copy dataset files from a local directory (e.g. USB or old clone)",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir

    if args.instructions:
        _print_instructions(data_dir)
        sys.exit(0)

    if args.copy_from:
        count = _copy_from_source(args.copy_from, data_dir)
        print(f"\nCopied {count} file(s).")
        sys.exit(_run_verify(data_dir))

    if args.extract_normal:
        ok = _extract_normal_archive(data_dir)
        sys.exit(0 if ok else 1)

    if args.verify or len(sys.argv) == 1:
        sys.exit(_run_verify(data_dir))

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
