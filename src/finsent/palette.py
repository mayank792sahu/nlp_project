"""Shared color system (validated categorical + status palette) used by both the
matplotlib report figures and the Streamlit dashboard, so the two stay visually
consistent instead of drifting to different default library colors."""

# Categorical, in fixed order — one color per model, never reassigned/cycled.
MODEL_COLORS = {
    "VADER": "#2a78d6",              # blue
    "Loughran-McDonald": "#eb6834",  # orange
    "TFIDF+ML": "#1baf7a",           # aqua
    "TF-IDF + ML": "#1baf7a",
    "FinBERT (zero-shot)": "#eda100",  # yellow
}

# Status palette — reserved for sentiment state, distinct from model identity colors.
SENTIMENT_COLORS = {
    "positive": "#0ca30c",  # good
    "negative": "#d03b3b",  # critical
    "neutral": "#52514e",   # secondary ink (no dedicated neutral status hue)
}

# Chart chrome / ink tokens (light surface).
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# Sequential blue ramp (light -> dark), for confusion-matrix heatmaps.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def model_color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, TEXT_SECONDARY)


def sentiment_color(label: str) -> str:
    return SENTIMENT_COLORS.get(label, TEXT_SECONDARY)
