"""Classifiers: a common interface, a rules baseline, and an LLM classifier.

- `Classifier` — the protocol everything implements.
- `HeuristicClassifier` — keyword/rule baseline. Needs no API key, so the
  service and the eval harness run end-to-end out of the box. Its eval scores
  are the baseline the LLM prompt has to beat.
- `LLMClassifier` — prompt-based classifier on top of any `LLMClient`.
  Weekend work: wire a real client (app/llm_client.py) and iterate the prompt
  against `python -m evals run` until the report is green.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol, runtime_checkable

from .llm_client import LLMClient, get_llm_client
from .models import ClassificationOut, EmailIn

# ---------------------------------------------------------------------------
# Shared extraction helpers (used by both classifiers)
# ---------------------------------------------------------------------------

# "Thank you for applying to Future", "your application to Cargomatic",
# "your application with Toast", "applying for the X position at Y" ...
_COMPANY_PATTERNS = [
    re.compile(r"applying to ([A-Z][A-Za-z0-9&.'\- ]{1,40}?)(?:[.,!|\n]|$)", re.M),
    re.compile(r"application (?:to|with|at) ([A-Z][A-Za-z0-9&.'\- ]{1,40}?)(?:[.,!|\n]|$)", re.M),
    re.compile(r"interest in ([A-Z][A-Za-z0-9&.'\- ]{1,40}?)(?:[.,!|\n]|$)", re.M),
    re.compile(r"welcome to ([A-Z][A-Za-z0-9&.'\- ]{1,40}?)(?:[.,!|\n]|$)", re.M),
]

# "for the Senior Software Engineer role", "for our Staff Engineer position"
_ROLE_PATTERNS = [
    re.compile(r"for the ([A-Z][A-Za-z0-9/&\- ,]{2,60}?) (?:role|position)(?:[.,!|\n]|$)", re.M),
    re.compile(r"for our ([A-Z][A-Za-z0-9/&\- ,]{2,60}?) (?:role|position)(?:[.,!|\n]|$)", re.M),
    re.compile(r"applying for ([A-Z][A-Za-z0-9/&\- ,]{2,60}?)(?:[.,!|\n]|$)", re.M),
]

# Generic words that pattern-matches above sometimes capture as a "company".
_NOT_A_COMPANY = {
    "us", "our team", "the team", "this role", "this position", "your",
    "the", "a", "an", "jobs", "careers", "talent",
}


def _clean_capture(text: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", text).strip(" .,!|:-")
    if not text or text.lower() in _NOT_A_COMPANY or len(text) < 2:
        return None
    return text


def extract_company(email: EmailIn) -> Optional[str]:
    text = f"{email.subject}\n{email.body}"
    for pattern in _COMPANY_PATTERNS:
        m = pattern.search(text)
        if m:
            company = _clean_capture(m.group(1))
            if company:
                return company
    return None


def extract_role(email: EmailIn) -> Optional[str]:
    text = f"{email.subject}\n{email.body}"
    for pattern in _ROLE_PATTERNS:
        m = pattern.search(text)
        if m:
            role = _clean_capture(m.group(1))
            if role:
                return role
    return None


# ---------------------------------------------------------------------------
# Classifier interface
# ---------------------------------------------------------------------------


@runtime_checkable
class Classifier(Protocol):
    name: str

    def classify(self, email: EmailIn) -> ClassificationOut:
        ...


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------

# (phrase, weight) — matched case-insensitively against subject + body.
_REJECTION_SIGNALS = [
    ("not moving forward", 3),
    ("move forward with other", 3),
    ("decided to move forward", 3),
    ("will not be moving forward", 3),
    ("unfortunately", 2),
    ("not selected", 3),
    ("not be selected", 3),
    ("we regret", 2),
    ("regret to inform", 3),
    ("other candidates", 1),
    ("not a fit", 2),
    ("not the right fit", 2),
    ("position has been filled", 3),
    ("no longer under consideration", 3),
    ("unable to move forward", 3),
    ("not proceed", 2),
    ("wish you the best", 1),
    ("future opportunities", 1),  # weak on its own; common in rejections
]

_CONFIRMATION_SIGNALS = [
    ("thank you for applying", 3),
    ("thanks for applying", 3),
    ("application received", 3),
    ("received your application", 3),
    ("we have received your application", 3),
    ("your application has been submitted", 3),
    ("application was successfully submitted", 3),
    ("thank you for your interest", 1),
    ("thank you for your application", 3),
    ("we will review your application", 2),
    ("we'll be in touch", 1),
    ("received your resume", 2),
    ("application confirmation", 3),
]

_INTERVIEW_SIGNALS = [
    ("schedule an interview", 3),
    ("schedule a call", 2),
    ("invite you to interview", 3),
    ("invitation to interview", 3),
    ("next steps", 1),
    ("phone screen", 3),
    ("speak with our team", 2),
    ("book a time", 2),
    ("pleased to invite", 3),
    ("move you to the next", 3),
    ("moving to the next stage", 3),
    ("congratulations", 1),  # weak: also appears in "congrats on applying"
]


def _score(text: str, signals: list[tuple[str, int]]) -> int:
    return sum(weight for phrase, weight in signals if phrase in text)


class HeuristicClassifier:
    """Keyword/rule baseline. Deterministic, free, and honest about it."""

    name = "heuristic"

    def classify(self, email: EmailIn) -> ClassificationOut:
        text = f"{email.subject}\n{email.body}".lower()
        scores = {
            "rejection": _score(text, _REJECTION_SIGNALS),
            "confirmation": _score(text, _CONFIRMATION_SIGNALS),
            "interview": _score(text, _INTERVIEW_SIGNALS),
        }
        best_class, best_score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        if best_score == 0:
            return ClassificationOut(
                **{
                    "class": "other",
                    "company": extract_company(email),
                    "role": extract_role(email),
                    "confidence": 0.3,
                }
            )
        # Confidence grows with the winner's share of total signal and with
        # absolute signal strength, capped so rules never claim certainty.
        share = best_score / total if total else 0.0
        confidence = min(0.95, 0.45 + 0.35 * share + 0.05 * best_score)
        return ClassificationOut(
            **{
                "class": best_class,
                "company": extract_company(email),
                "role": extract_role(email),
                "confidence": round(confidence, 3),
            }
        )


# ---------------------------------------------------------------------------
# LLM classifier (prompt scaffold — needs a real client to run)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You classify job-application emails.

Given the email below, reply with ONLY a JSON object:
{{"class": "rejection" | "confirmation" | "interview" | "other",
  "company": <string or null>,
  "role": <string or null>,
  "confidence": <number between 0 and 1>}}

Definitions:
- rejection: the company is not moving forward with the application.
- confirmation: receipt of the application was confirmed; no decision yet.
- interview: the company is inviting the candidate to a call/interview or
  explicitly moving them to a next stage.
- other: anything else (alerts, marketing, unrelated).

From: {from_}
Subject: {subject}

{body}
"""


class LLMClassifier:
    """Prompt-based classifier over any LLMClient.

    Weekend TODO: iterate PROMPT_TEMPLATE against the eval harness
    (`python -m evals run --classifier llm`) until per-class precision/recall
    beat the heuristic baseline.
    """

    name = "llm"

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client or get_llm_client()

    def classify(self, email: EmailIn) -> ClassificationOut:
        prompt = PROMPT_TEMPLATE.format(
            from_=email.from_, subject=email.subject, body=email.body[:4000]
        )
        raw = self.client.complete(prompt)
        return parse_llm_response(raw, email)


def parse_llm_response(raw: str, email: EmailIn) -> ClassificationOut:
    """Parse the model's JSON reply, falling back to extraction helpers."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"LLM response contained no JSON object: {raw[:200]!r}")
    data = json.loads(match.group(0))
    cls = data.get("class", "other")
    if cls not in ("rejection", "confirmation", "interview", "other"):
        cls = "other"
    return ClassificationOut(
        **{
            "class": cls,
            "company": data.get("company") or extract_company(email),
            "role": data.get("role") or extract_role(email),
            "confidence": float(data.get("confidence", 0.5)),
        }
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_classifier(name: Optional[str] = None) -> Classifier:
    """Build the classifier named by `name` or the CLASSIFIER env var."""
    chosen = (name or os.environ.get("CLASSIFIER", "heuristic")).strip().lower()
    if chosen == "heuristic":
        return HeuristicClassifier()
    if chosen == "llm":
        return LLMClassifier()
    raise ValueError(f"Unknown classifier {chosen!r} (expected 'heuristic' or 'llm')")
