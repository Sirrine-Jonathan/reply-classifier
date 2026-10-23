"""Pydantic models for the Reply Classifier API.

Request/response schemas live here so the service, the eval harness, and any
future client share one definition of an email and a classification.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# The four classes the service can return. The gold set currently covers
# "rejection" and "confirmation"; "interview" and "other" are first-class so
# the classifier never has to force an email into a wrong bucket.
EmailClass = Literal["rejection", "confirmation", "interview", "other"]


class EmailIn(BaseModel):
    """One email to classify."""

    model_config = ConfigDict(populate_by_name=True)

    subject: str = ""
    body: str = ""
    from_: str = Field(default="", alias="from")


class ClassificationOut(BaseModel):
    """Classification result for one email."""

    classification: EmailClass = Field(..., alias="class")
    company: Optional[str] = None
    role: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(populate_by_name=True)


class BatchIn(BaseModel):
    emails: list[EmailIn]


class BatchOut(BaseModel):
    results: list[ClassificationOut]


class HealthOut(BaseModel):
    status: str = "ok"
    classifier: str
