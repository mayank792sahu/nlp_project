"""Stratified train/val/test split, computed once and cached to data/processed/."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from finsent.config import CONFIG, resolve_path


def _cache_paths() -> dict:
    cache_dir = resolve_path(CONFIG["splits"]["cache_dir"])
    return {name: cache_dir / f"{name}.csv" for name in ("train", "val", "test")}


def get_splits(df: pd.DataFrame, force_reload: bool = False) -> dict:
    """Returns {'train': df, 'val': df, 'test': df}, stratified on 'label'."""
    paths = _cache_paths()

    if not force_reload and all(p.exists() for p in paths.values()):
        return {name: pd.read_csv(p) for name, p in paths.items()}

    cfg = CONFIG["splits"]
    seed = CONFIG["seed"]

    train_df, temp_df = train_test_split(
        df, test_size=(1 - cfg["train"]), stratify=df["label"], random_state=seed
    )
    val_ratio_of_temp = cfg["val"] / (cfg["val"] + cfg["test"])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp), stratify=temp_df["label"], random_state=seed
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(paths[name], index=False)

    return splits
