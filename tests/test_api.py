"""Smoke tests for the FastAPI service (heuristic classifier, no API key)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REJECTION = {
    "subject": "Your application for Grafana Labs",
    "from": "no-reply@grafana.com",
    "body": (
        "Thank you for applying for the Staff Frontend Engineer role. "
        "Unfortunately, we have decided to move forward with other candidates. "
        "We wish you the best in your search."
    ),
}

CONFIRMATION = {
    "subject": "Thank you for applying to Future",
    "from": "no-reply@us.greenhouse-mail.io",
    "body": (
        "Thank you for applying to Future. Your application has been received "
        "and we will review your application shortly."
    ),
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["classifier"]  # name of the active classifier is reported


def test_classify_rejection_shape():
    resp = client.post("/classify", json=REJECTION)
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload) == {"class", "company", "role", "confidence"}
    assert payload["class"] == "rejection"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_classify_confirmation():
    resp = client.post("/classify", json=CONFIRMATION)
    assert resp.status_code == 200
    assert resp.json()["class"] == "confirmation"


def test_classify_batch():
    resp = client.post("/classify/batch", json={"emails": [REJECTION, CONFIRMATION]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["class"] == "rejection"
    assert results[1]["class"] == "confirmation"
