from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"


class TicketPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run training/train.py first."
            )
        self.model = joblib.load(MODEL_PATH)

    def predict(self, message: str) -> tuple[str, float]:
        probabilities = self.model.predict_proba([message])[0]
        predicted_index = probabilities.argmax()
        predicted_label = self.model.classes_[predicted_index]
        confidence = float(probabilities[predicted_index])

        return predicted_label, confidence