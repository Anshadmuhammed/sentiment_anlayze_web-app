"""
train.py  –  Train and save the Sentiment Analysis model.
Run this once before launching app.py:
    python train.py
"""

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Sample training data ───────────────────────────────────────────────────────
# Replace or extend this with a real dataset (e.g. IMDB, Amazon reviews, Twitter)
TEXTS = [
    # Positive
    "This product is absolutely amazing, I love it!",
    "Great quality, fast delivery, very happy with my purchase.",
    "Excellent service, the staff were very helpful and friendly.",
    "Totally worth the money, highly recommend to everyone.",
    "The best experience I've ever had, will definitely come back.",
    "Outstanding performance, exceeded all my expectations.",
    "Wonderful product, works perfectly right out of the box.",
    "Very satisfied with the quality and overall value.",
    "Incredible results, couldn't be happier with this.",
    "Superb craftsmanship, arrived in perfect condition.",
    # Negative
    "Terrible product, broke after just one day of use.",
    "Very disappointed, this was a complete waste of money.",
    "Awful customer service, nobody helped me resolve the issue.",
    "Poor quality, the item looks nothing like the pictures.",
    "I hate this product, it doesn't work at all.",
    "Horrible experience, I will never buy from here again.",
    "Complete garbage, fell apart immediately.",
    "The worst purchase I have ever made, deeply regret it.",
    "Very bad quality control, arrived damaged.",
    "Dreadful performance, stopped working within a week.",
    # Neutral
    "The product is okay, nothing special but it does the job.",
    "It arrived on time and works as described.",
    "Average quality for the price, not good but not bad.",
    "It's a standard product, does what it says on the tin.",
    "Delivery was fine, product is acceptable.",
    "Works as expected, no complaints but no excitement either.",
    "It's alright, I've seen better but also worse.",
    "Reasonable product, meets basic requirements.",
    "Not bad, not great, just average overall.",
    "It functions correctly, nothing more nothing less.",
]

LABELS = (
    ["Positive"] * 10 +
    ["Negative"] * 10 +
    ["Neutral"]  * 10
)

# ── Build & train pipeline ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    TEXTS, LABELS, test_size=0.2, random_state=42, stratify=LABELS
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ── Save model ────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved to model.pkl")
print("Run:  streamlit run app.py")
