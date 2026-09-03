"""Loads the Financial PhraseBank dataset, caching a normalized CSV to data/raw/.

Primary path: Hugging Face `datasets` (takala/financial_phrasebank).
Fallback path: manual parse of a raw `SentencesXXAgree.txt` file (sentence@label, latin-1),
used if the `datasets` download fails (e.g. no network access).
"""

import logging
from pathlib import Path

import pandas as pd

from finsent.config import CONFIG, resolve_path
from finsent.labels import PHRASEBANK_ID2LABEL

logger = logging.getLogger(__name__)


def _load_from_hf(hf_name: str, hf_config: str = None) -> pd.DataFrame:
    """Loads via the Hugging Face `datasets` library.

    The canonical `takala/financial_phrasebank` repo uses a loading-script format that
    recent `datasets` versions (>=4.0) refuse to execute ("Dataset scripts are no longer
    supported"), and it has no auto-converted parquet revision either. `hf_name` should
    therefore point at a script-free parquet mirror (see config.yaml) — `hf_config` is
    only passed through for repos that expose multiple configs/subsets.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)

    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    df = df.rename(columns={"sentence": "sentence", "label": "label_id"})
    df["label"] = df["label_id"].map(PHRASEBANK_ID2LABEL)
    return df[["sentence", "label"]]


def _load_from_manual_file(file_path: Path) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
    return pd.DataFrame(rows)


def load_financial_phrasebank(force_reload: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with columns ['sentence', 'label'], caching to data/raw/."""
    ds_cfg = CONFIG["dataset"]
    cache_path = resolve_path(ds_cfg["cache_csv"])

    if cache_path.exists() and not force_reload:
        return pd.read_csv(cache_path)

    df = None
    try:
        df = _load_from_hf(ds_cfg["hf_name"], ds_cfg.get("hf_config"))
        logger.info("Loaded Financial PhraseBank via Hugging Face datasets.")
    except Exception as e:
        logger.warning("HF datasets load failed (%s); falling back to manual file.", e)
        fallback_path = resolve_path(ds_cfg["fallback_file"])
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"HF datasets load failed and fallback file not found at {fallback_path}. "
                "Download FinancialPhraseBank-v1.0.zip and place the relevant "
                "SentencesXXAgree.txt file there."
            ) from e
        df = _load_from_manual_file(fallback_path)
        logger.info("Loaded Financial PhraseBank via manual fallback file.")

    df = df.dropna(subset=["sentence", "label"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
