# Financial Sentiment Classification — Comparative Study

Lexicon-based (VADER, Loughran-McDonald) vs. machine learning (TF-IDF + Logistic Regression/SVM) vs. pretrained transformer (FinBERT, zero-shot) sentiment classification on the Financial PhraseBank dataset. Runs entirely on CPU — no GPU required.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

One-time manual steps:

1. **NLTK data** (or just run `notebooks/report.ipynb` / `pytest`, both trigger this automatically):
   ```python
   import nltk
   nltk.download(["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"])
   ```
2. **Loughran-McDonald dictionary** — not distributed via pip. Download the Master Dictionary CSV from the [Notre Dame SRAF site](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and save it to `data/external/LM_MasterDictionary.csv`.
3. **Financial PhraseBank** — downloads automatically via Hugging Face `datasets` (`warwickai/financial_phrasebank_mirror`) on first run and caches to `data/raw/`. This mirror holds the full ~4,846-sentence set with all annotator-agreement levels merged (the canonical `takala/financial_phrasebank` repo no longer loads under current `datasets` versions — see the comment in `config.yaml`). For the cleaner `sentences_66agree`/`allagree` subsets instead, download `FinancialPhraseBank-v1.0.zip` manually and place `Sentences66Agree.txt` at `data/external/Sentences66Agree.txt`.

## Usage

Open `notebooks/report.ipynb` and run all cells — it loads the data, runs all four models on the same held-out test split, and writes the comparison table/figures to `results/`.

```bash
pytest
```
