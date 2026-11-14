"""End-to-end RAG pipeline: retrieve -> generate -> (optionally) evaluate."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd

from src.config import Settings, get_settings
from src.evaluate import METRICS, evaluate_records
from src.generator import generate_answer
from src.history import log_run
from src.retriever import RetrievedChunk, retrieve


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    ground_truth: str = ""
    scores: dict | None = None
    latency_s: float | None = None
    error: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": [chunk.to_dict() for chunk in self.contexts],
            "ground_truth": self.ground_truth,
            "scores": self.scores,
            "latency_s": self.latency_s,
        }

    def to_record(self) -> dict:
        """Shape expected by `evaluate_records`."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": [chunk.content for chunk in self.contexts],
            "ground_truth": self.ground_truth,
        }


def answer_question(
    question: str,
    top_k: int | None = None,
    search_type: str | None = None,
    ground_truth: str = "",
    settings: Settings | None = None,
    vectorstore=None,
    llm=None,
) -> RAGResult:
    """Retrieve context and generate an answer (no evaluation)."""
    settings = settings or get_settings()
    started = time.perf_counter()
    chunks = retrieve(question, top_k, search_type, settings, vectorstore)
    answer = generate_answer(question, chunks, settings, llm)
    return RAGResult(
        question=question.strip(),
        answer=answer,
        contexts=chunks,
        ground_truth=(ground_truth or "").strip(),
        latency_s=round(time.perf_counter() - started, 2),
    )


def evaluate_result(result: RAGResult, settings: Settings | None = None, runner=None) -> RAGResult:
    result.scores = evaluate_records([result.to_record()], settings, runner)[0]
    return result


def run_batch(
    rows: Sequence[dict],
    top_k: int | None = None,
    search_type: str | None = None,
    settings: Settings | None = None,
    vectorstore=None,
    llm=None,
    runner=None,
    progress: Callable[[int, int], None] | None = None,
    log: bool = True,
) -> pd.DataFrame:
    """Answer and evaluate a test set. Each row needs `question`; `ground_truth` optional."""
    settings = settings or get_settings()
    results: list[RAGResult] = []
    for number, row in enumerate(rows, start=1):
        question = str(row.get("question") or "").strip()
        truth = str(row.get("ground_truth") or "").strip()
        try:
            result = answer_question(
                question, top_k, search_type, truth, settings, vectorstore, llm
            )
        except Exception as exc:  # keep going; surface the failure in the report
            result = RAGResult(question, "", [], truth, error=f"{type(exc).__name__}: {exc}")
        results.append(result)
        if progress:
            progress(number, len(rows))

    scores = evaluate_records([r.to_record() for r in results], settings, runner)
    table = []
    for result, score in zip(results, scores):
        result.scores = score
        if log and not result.error:
            log_run(settings.history_file, result.to_dict())
        table.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth,
                "sources": ", ".join(sorted({c.source for c in result.contexts})),
                **{metric: score.get(metric) for metric in METRICS},
                "latency_s": result.latency_s,
                "error": result.error,
            }
        )
    return pd.DataFrame(table)


def parse_test_set(frame: pd.DataFrame) -> list[dict]:
    """Validate an uploaded test set and return rows with `question`/`ground_truth`."""
    frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    if "question" not in frame.columns:
        raise ValueError("The file needs a 'question' column (and optionally 'ground_truth').")
    if "ground_truth" not in frame.columns:
        frame["ground_truth"] = ""
    frame = frame[["question", "ground_truth"]].fillna("").astype(str)
    frame = frame[frame["question"].str.strip() != ""]
    if frame.empty:
        raise ValueError("The file contains no questions.")
    return frame.to_dict("records")
