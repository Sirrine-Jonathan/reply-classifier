# Eval report — classifier: `heuristic`

- Dataset: `/home/hatch/workspace/reply-classifier/evals/data/gold_set.jsonl` (147 emails)
- Overall accuracy: **95.9%**
- Verdict: **PASS** (bar: accuracy >= 90%, recall >= 80% per class with support >= 5)

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| rejection | 29 | 1.000 | 0.931 | 0.964 |
| confirmation | 118 | 0.991 | 0.966 | 0.979 |
| interview | 0 | 0.000 | 0.000 | 0.000 |
| other | 0 | 0.000 | 0.000 | 0.000 |

## Confusion (gold -> predicted)

| gold \ pred | rejection | confirmation | interview | other |
|---|---|---|---|---|
| rejection | 27 | 1 | 0 | 1 |
| confirmation | 0 | 114 | 2 | 2 |

## Extraction

- company: not evaluated (no annotated gold records)
- role: not evaluated (no annotated gold records)
