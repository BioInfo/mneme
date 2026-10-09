"""Configuration loading for Mneme."""

import os
from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in project root
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand ~ in paths
    for source in config.get("sources", []):
        source["path"] = str(Path(source["path"]).expanduser())

    if "vectordb" in config:
        config["vectordb"]["path"] = str(
            Path(config["vectordb"]["path"]).expanduser()
        )

    # Per-process device overrides, so the API can run the reranker on CUDA and
    # the embedder on CPU while the hourly indexer, same config file, keeps CUDA
    # for bulk embedding. Set in the mneme-api unit, nowhere else.
    if os.environ.get("MNEME_EMBED_DEVICE"):
        config.setdefault("embeddings", {})["device"] = os.environ["MNEME_EMBED_DEVICE"]
    if os.environ.get("MNEME_RERANK_DEVICE"):
        config.setdefault("rerank", {})["device"] = os.environ["MNEME_RERANK_DEVICE"]

    return config
