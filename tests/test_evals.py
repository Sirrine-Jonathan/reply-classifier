"""Smoke tests for the eval harness on the tiny hand-labeled fixture."""

from pathlib import Path

from app.classifier import HeuristicClassifier
from evals.export_gold_set import clean_body
from evals.runner import load_gold_set, run_eval

FIXTURE = Path(__file__).resolve().parent.parent / "evals" / "data" / "fixture_tiny.jsonl"


def test_eval_runner_on_fixture():
    records = load_gold_set(FIXTURE)
    assert len(records) == 6
    report = run_eval(HeuristicClassifier(), records, dataset_name=str(FIXTURE))
    assert report.total == 6
    assert 0.0 <= report.accuracy <= 1.0
    # Every gold class with examples shows up in the per-class table.
    for cls in ("rejection", "confirmation", "interview"):
        assert cls in report.per_class
        assert report.per_class[cls]["support"] >= 1
    # The fixture annotates company on two records, so extraction is evaluated.
    assert report.extraction["company"]["evaluated"] == 2
    markdown = report.to_markdown()
    assert "Precision" in markdown and "Confusion" in markdown


def test_clean_body_strips_quoted_chain():
    raw = (
        "Thanks for applying!\n"
        "> On Tuesday, someone wrote:\n"
        "> the original message\n"
        "On Wed, Oct 22, 2026 at 9:00 AM X <x@example.com> wrote:\n"
        "older quoted text\n"
    )
    cleaned = clean_body(raw)
    assert "Thanks for applying!" in cleaned
    assert "original message" not in cleaned
    assert "older quoted text" not in cleaned
