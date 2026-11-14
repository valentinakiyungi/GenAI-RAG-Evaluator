"""Append-only log of questions, answers and evaluation scores (JSON Lines)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluate import METRICS

COLUMNS = ["timestamp", "question", "answer", "ground_truth", "sources", *METRICS, "latency_s"]


def log_run(path: Path, result: dict) -> None:
    """Append one pipeline result (see `RAGResult.to_dict`) to the history file."""
    scores = result.get("scores") or {}
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": result.get("question", ""),
        "answer": result.get("answer", ""),
        "ground_truth": result.get("ground_truth", ""),
        "sources": sorted({c.get("source", "") for c in result.get("contexts", [])}),
        **{metric: scores.get(metric) for metric in METRICS},
        "latency_s": result.get("latency_s"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history(path: Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than lose the whole log
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for metric in (*METRICS, "latency_s"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def clear_history(path: Path) -> None:
    Path(path).unlink(missing_ok=True)
