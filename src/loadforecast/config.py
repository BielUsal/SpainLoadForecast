"""Project paths and YAML config loading."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
