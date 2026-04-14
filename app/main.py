from fastapi import FastAPI
from app.predictor import TicketPredictor
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="Support Ticket Classifier API", version="0.1.0")

predictor = TicketPredictor()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    label, confidence = predictor.predict(request.message)
    return PredictionResponse(predicted_label=label, confidence=confidence)