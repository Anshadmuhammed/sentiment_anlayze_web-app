"""
predict.py  –  Load the saved model and predict sentiment.
"""

import pickle
import os
import re
import string

MODEL_PATH = "model.pkl"

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # normalise whitespace
    return text

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. Please run  `python train.py`  first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

_model = None

def predict_sentiment(text: str):
    """
    Returns (label, confidence_percent).
    label       : 'Positive' | 'Negative' | 'Neutral'
    confidence  : float 0-100
    """
    global _model
    if _model is None:
        _model = load_model()

    cleaned = preprocess(text)
    label   = _model.predict([cleaned])[0]
    proba   = _model.predict_proba([cleaned])[0]
    confidence = max(proba) * 100
    return label, confidence
