#!/usr/bin/env python3
"""Export the eval gold set from Jonathan's labeled Gmail messages.

Pulls every message under the "Job Applications/Rejections" label (Label_7)
and the "Job Applications/Confirmations" label (Label_8) via the Gmail
connector CLI and writes them as JSONL:

    {"id", "subject", "from", "body", "label"}   label: rejection|confirmation

The output (evals/data/gold_set.jsonl) is PRIVATE local data — Jonathan's own
mail, for his own local eval harness. It is gitignored; never upload it.

Usage:
    python evals/export_gold_set.py [--limit N] [--out PATH]

Requires the Gmail connector to be connected (`hatch_gws_cli gmail status`).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

GWS_CLI = os.environ.get("GWS_CLI", "/opt/hatch/bin/hatch_gws_cli")

# Gmail label ID -> gold-set label. Rejections are exported first so that, in
# the unlikely case a message carries both labels, "rejection" wins the dedupe
# (a rejection that also confirms receipt is still a rejection).
LABELS = [
    ("Label_7", "rejection"),       # Job Applications/Rejections
    ("Label_8", "confirmation"),    # Job Applications/Confirmations
]

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "gold_set.jsonl"
MAX_BODY_CHARS = 4000


def run_cli(args: list[str], retries: int = 3) -> str:
    """Run one Gmail CLI command, retrying transient transport failures."""
    last_error = ""
    for attempt in range(1, retries + 1):
        proc = subprocess.run(
            [GWS_CLI, "gmail", *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return proc.stdout
        last_error = proc.stderr.strip()
        if attempt < retries:
            print(
                f"  CLI attempt {attempt} failed, retrying: {last_error[:120]}",
                file=sys.stderr,
            )
            time.sleep(2 * attempt)
    raise RuntimeError(f"CLI failed ({args[:3]}...): {last_error[:300]}")


def parse_json_stream(text: str) -> list[dict]:
    """Parse one or more concatenated JSON values (the CLI's --page-all mode
    prints one JSON object per page)."""
    decoder = json.JSONDecoder()
    out: list[dict] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        out.append(obj)
        idx = end
    return out


def list_message_ids(label_id: str) -> list[str]:
    params = json.dumps(
        {"userId": "me", "labelIds": [label_id], "maxResults": 500}
    )
    stdout = run_cli(["users", "messages", "list", "--params", params, "--page-all"])
    ids: list[str] = []
    for page in parse_json_stream(stdout):
        for message in page.get("messages", []):
            ids.append(message["id"])
    return ids


def _decode_body_data(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _collect_parts(part: dict, plain: list[str], html_parts: list[str]) -> None:
    mime = part.get("mimeType", "")
    body = part.get("body") or {}
    data = body.get("data")
    if data:
        text = _decode_body_data(data)
        if mime == "text/plain":
            plain.append(text)
        elif mime == "text/html":
            html_parts.append(text)
    for sub in part.get("parts", []) or []:
        _collect_parts(sub, plain, html_parts)


def _html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)</p>", "\n\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


_QUOTE_MARKERS = (
    re.compile(r"^On .+ wrote:$"),
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.I),
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I),
)


def clean_body(text: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Drop quoted reply chains (best effort) and truncate."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(p.search(stripped) for p in _QUOTE_MARKERS):
            break
        if stripped.startswith(">"):
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned[:max_chars]


def fetch_message(message_id: str) -> dict:
    params = json.dumps({"userId": "me", "id": message_id, "format": "full"})
    stdout = run_cli(["users", "messages", "get", "--params", params])
    pages = parse_json_stream(stdout)
    message = pages[0] if pages else {}
    payload = message.get("payload", {})
    headers = {
        h.get("name", ""): h.get("value", "") for h in payload.get("headers", [])
    }
    plain: list[str] = []
    html_parts: list[str] = []
    _collect_parts(payload, plain, html_parts)
    if plain:
        body = "\n".join(plain)
    elif html_parts:
        body = _html_to_text("\n".join(html_parts))
    else:
        body = message.get("snippet", "")
    return {
        "id": message_id,
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "body": clean_body(body),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the eval gold set from Gmail labels")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max messages per label (for a quick smoke test)",
    )
    args = parser.parse_args(argv)

    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for label_id, gold_label in LABELS:
        ids = list_message_ids(label_id)
        if args.limit:
            ids = ids[: args.limit]
        print(f"{gold_label}: {len(ids)} message(s) under {label_id}", file=sys.stderr)
        fetched = 0
        for i, message_id in enumerate(ids, 1):
            if message_id in seen:
                print(f"  note: {message_id} already exported; skipping duplicate",
                      file=sys.stderr)
                continue
            record = fetch_message(message_id)
            record["label"] = gold_label
            seen[message_id] = record
            fetched += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(ids)} fetched", file=sys.stderr)
        counts[gold_label] = fetched

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(seen.values(), key=lambda r: (r["label"], r["id"]))
    with args.out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
