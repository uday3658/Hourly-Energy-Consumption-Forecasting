"""
Utility to load config/config.yaml into a simple dict/object.
"""

import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the YAML config file and return it as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
