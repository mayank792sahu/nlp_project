"""Loads config.yaml and resolves paths relative to the project root."""

import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


CONFIG = load_config()
set_seed(CONFIG["seed"])
