"""FastAPI app for the Reply Classifier service.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    GET  /health           -> {"status": "ok", "classifier": "<name>"}
    POST /classify         -> one email  -> one classification
    POST /classify/batch   -> many emails -> many classifications

The active classifier is chosen by the CLASSIFIER env var
("heuristic" by default — no API key needed; "llm" once a provider is
wired into app/llm_client.py).
"""

from __future__ import annotations

from fastapi import FastAPI

from .classifier import Classifier, get_classifier
from .models import BatchIn, BatchOut, ClassificationOut, EmailIn, HealthOut

app = FastAPI(
    title="Reply Classifier",
    description="Classifies job-application emails: rejection / confirmation / interview / other.",
    version="0.1.0",
)

_classifier: Classifier = get_classifier()


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", classifier=_classifier.name)


@app.post("/classify", response_model=ClassificationOut, response_model_by_alias=True)
def classify(email: EmailIn) -> ClassificationOut:
    return _classifier.classify(email)


@app.post("/classify/batch", response_model=BatchOut, response_model_by_alias=True)
def classify_batch(batch: BatchIn) -> BatchOut:
    return BatchOut(results=[_classifier.classify(email) for email in batch.emails])
