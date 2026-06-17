"""Graph dataset caching and loading."""

from pathlib import Path

import torch
from chebnet_kaist.data.cache.paths import resolve_readable_cache
from chebnet_kaist.data.graphs.builder import build_graph_dataset
from chebnet_kaist.data.preprocessing.segments import iter_all_segments
from torch_geometric.data import Data


def load_or_build_graphs(
    data_dir: str | Path,
    cache_file: str | Path,
    segment_size: float = 0.1,
    ignore_cache: bool = False,
) -> list[Data]:
    """Load pre-built graphs from cache or build them from raw CAN data.

    Args:
        data_dir: Root directory of the KAIST dataset.
        cache_file: Path to the ``.pt`` cache file.
        segment_size: Sliding segment size in seconds.
        ignore_cache: When True, rebuild graphs even if cache exists.

    Returns:
        List of PyG graph objects.
    """
    cache_path = resolve_readable_cache(cache_file)

    if cache_path.exists() and not ignore_cache:
        print(f"Cache found: {cache_path}")
        print("Loading graphs from disk...")
        graphs: list[Data] = torch.load(cache_path, weights_only=False)
        print(f"  Loaded {len(graphs):,} graphs.")
        return graphs

    print("Building graph dataset (each class processed separately)...")
    segments = iter_all_segments(data_dir, segment_size)
    graphs = build_graph_dataset(segments, show_progress=True)
    print(f"  Built {len(graphs):,} graphs total.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cache: {cache_path}")
    torch.save(graphs, cache_path)
    return graphs
