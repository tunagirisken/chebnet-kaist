"""Graph cache path resolution with legacy fallbacks."""

from pathlib import Path


def resolve_readable_cache(cache_file: str | Path) -> Path:
    """Return an existing cache file path, including legacy Turkish filename.

    Args:
        cache_file: Preferred cache path from experiment config.

    Returns:
        ``cache_file`` when it exists, otherwise ``graflar_cache.pt`` in the
        same directory when that legacy file exists, else the original path.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        return cache_path
    legacy_cache = cache_path.parent / "graflar_cache.pt"
    if legacy_cache.exists():
        return legacy_cache
    return cache_path
