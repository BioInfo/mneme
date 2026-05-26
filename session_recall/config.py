"""Configuration loading for Session Recall."""

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

    return config
