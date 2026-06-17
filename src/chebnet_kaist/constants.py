"""Project-wide constants shared across modules."""

CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "RPM Spoofing", "Gear Spoofing"]
CLASS_NAMES_SHORT = ["normal", "dos", "fuzzy", "rpm", "gear"]
BINARY_CLASS_NAMES_SHORT = ["normal", "attack"]
BINARY_CLASS_NAMES = ["Normal", "Attack"]
MODEL_LABELS = {
    "chebnet": "ChebNetIDS",
}

EMBEDDING_VIZ_MODELS = frozenset({"chebnet"})
