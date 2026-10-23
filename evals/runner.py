"""Eval runner: score a classifier against the labeled gold set.

Rubric (from the weekend spec):
- Class accuracy overall, plus per-class precision / recall / F1.
- Company / role extraction correctness, measured over the gold records that
  carry an annotated `company` / `role` (the Gmail export leaves them
  unannotated, so extraction is reported as "not evaluated" until a subset is
  hand-labeled — see README).

PASS bar (defaults, recorded in every report):
- overall accuracy >= 0.90
- recall >= 0.80 for every class with gold support >= 5

Reports are written as JSON (machine-readable) and Markdown (human-readable)
under reports/ and printed to stdout.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.classifier import Classifier
from app.models import EmailIn

ALL_CLASSES = ["rejection", "confirmation", "interview", "other"]


@dataclass
class GoldRecord:
    id: str
    subject: str
    from_: str
    body: str
    label: str
    company: Optional[str] = None
    role: Optional[str] = None

    def to_email(self) -> EmailIn:
        return EmailIn(**{"subject": self.subject, "body": self.body, "from": self.from_})


def load_gold_set(path: Path) -> list[GoldRecord]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                GoldRecord(
                    id=raw["id"],
                    subject=raw.get("subject", ""),
                    from_=raw.get("from", ""),
                    body=raw.get("body", ""),
                    label=raw["label"],
                    company=raw.get("company"),
                    role=raw.get("role"),
                )
            )
    return records


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class EvalReport:
    classifier: str
    dataset: str
    total: int
    accuracy: float
    per_class: dict[str, dict[str, float | int]]
    confusion: dict[str, dict[str, int]]
    extraction: dict[str, dict[str, float | int | None]]
    passed: bool
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier,
            "dataset": self.dataset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "per_class": self.per_class,
            "confusion": self.confusion,
            "extraction": self.extraction,
            "thresholds": self.thresholds,
            "passed": self.passed,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Eval report — classifier: `{self.classifier}`",
            "",
            f"- Dataset: `{self.dataset}` ({self.total} emails)",
            f"- Overall accuracy: **{self.accuracy:.1%}**",
            f"- Verdict: **{'PASS' if self.passed else 'FAIL'}** "
            f"(bar: accuracy >= {self.thresholds['min_accuracy']:.0%}, "
            f"recall >= {self.thresholds['min_recall']:.0%} per class with support >= 5)",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for cls in ALL_CLASSES:
            m = self.per_class.get(cls)
            if not m:
                continue
            lines.append(
                f"| {cls} | {m['support']} | {m['precision']:.3f} | "
                f"{m['recall']:.3f} | {m['f1']:.3f} |"
            )
        lines += ["", "## Confusion (gold -> predicted)", ""]
        lines.append("| gold \\ pred | " + " | ".join(ALL_CLASSES) + " |")
        lines.append("|" + "---|" * (len(ALL_CLASSES) + 1))
        for gold in ALL_CLASSES:
            row = self.confusion.get(gold, {})
            if not row:
                continue
            lines.append(
                f"| {gold} | " + " | ".join(str(row.get(p, 0)) for p in ALL_CLASSES) + " |"
            )
        lines += ["", "## Extraction", ""]
        for field_name, stats in self.extraction.items():
            if stats["evaluated"] == 0:
                lines.append(f"- {field_name}: not evaluated (no annotated gold records)")
            else:
                lines.append(
                    f"- {field_name}: {stats['accuracy']:.1%} correct "
                    f"over {stats['evaluated']} annotated records"
                )
        lines.append("")
        return "\n".join(lines)


def run_eval(
    classifier: Classifier,
    records: list[GoldRecord],
    dataset_name: str,
    min_accuracy: float = 0.90,
    min_recall: float = 0.80,
) -> EvalReport:
    confusion: dict[str, Counter] = defaultdict(Counter)
    extraction_hits = {"company": 0, "role": 0}
    extraction_total = {"company": 0, "role": 0}
    correct = 0

    for record in records:
        pred = classifier.classify(record.to_email())
        confusion[record.label][pred.classification] += 1
        if pred.classification == record.label:
            correct += 1
        for field_name, gold_value, pred_value in (
            ("company", record.company, pred.company),
            ("role", record.role, pred.role),
        ):
            if gold_value:
                extraction_total[field_name] += 1
                if _norm(gold_value) == _norm(pred_value):
                    extraction_hits[field_name] += 1

    total = len(records)
    per_class: dict[str, dict[str, float | int]] = {}
    for cls in ALL_CLASSES:
        tp = confusion[cls][cls]
        support = sum(confusion[cls].values())
        predicted = sum(confusion[gold][cls] for gold in ALL_CLASSES)
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if support or predicted:
            per_class[cls] = {
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    accuracy = correct / total if total else 0.0
    passed = accuracy >= min_accuracy and all(
        m["recall"] >= min_recall
        for m in per_class.values()
        if isinstance(m["support"], int) and m["support"] >= 5
    )
    extraction = {
        name: {
            "evaluated": extraction_total[name],
            "accuracy": (
                round(extraction_hits[name] / extraction_total[name], 4)
                if extraction_total[name]
                else None
            ),
        }
        for name in ("company", "role")
    }
    return EvalReport(
        classifier=classifier.name,
        dataset=dataset_name,
        total=total,
        accuracy=accuracy,
        per_class=per_class,
        confusion={gold: dict(counts) for gold, counts in confusion.items()},
        extraction=extraction,
        passed=passed,
        thresholds={"min_accuracy": min_accuracy, "min_recall": min_recall},
    )


def save_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"eval-{report.classifier}-{stamp}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path
