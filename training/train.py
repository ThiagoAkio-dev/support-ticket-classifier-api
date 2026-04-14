from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    X = df["text"]
    y = df["label"]

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()