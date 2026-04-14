from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=2000)


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float