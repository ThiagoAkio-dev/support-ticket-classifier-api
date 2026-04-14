from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict() -> None:
    response = client.post(
        "/predict",
        json={"message": "I was charged twice for my subscription"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert "predicted_label" in payload
    assert "confidence" in payload
    assert isinstance(payload["confidence"], float)