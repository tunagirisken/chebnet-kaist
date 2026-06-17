"""KAIST Car-Hacking graph dataset provider."""

from chebnet_kaist.config.schema import ExperimentConfig
from chebnet_kaist.constants import BINARY_CLASS_NAMES_SHORT, CLASS_NAMES_SHORT
from chebnet_kaist.data.cache.kaist import load_or_build_graphs
from torch_geometric.data import Data


class KaistGraphProvider:
    """Build or load graphs from the KAIST Car-Hacking dataset."""

    @property
    def name(self) -> str:
        """Registry key."""
        return "kaist"

    def class_names(self, label_mode: str = "multiclass") -> list[str]:
        """Return KAIST class labels."""
        if label_mode == "binary":
            return BINARY_CLASS_NAMES_SHORT
        return CLASS_NAMES_SHORT

    def load_graphs(self, config: ExperimentConfig) -> list[Data]:
        """Load the full KAIST graph cache for training or evaluation."""
        return load_or_build_graphs(
            data_dir=config.paths.data_dir,
            cache_file=config.paths.cache_file,
            segment_size=config.segment_size_sec,
            ignore_cache=config.training.ignore_cache,
        )


KAIST_PROVIDER = KaistGraphProvider()
