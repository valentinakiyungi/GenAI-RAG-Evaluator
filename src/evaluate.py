"""RAGAS evaluation: faithfulness, answer relevancy and context precision.

Context precision is computed against the reference answer when one is given,
otherwise against the generated response (reference-free variant).
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from src.config import Settings, get_settings, require_api_key

METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def _clean(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, 4)


def extract_scores(frame: pd.DataFrame) -> list[dict[str, float | None]]:
    """Normalise a RAGAS result frame into one {metric: score} dict per row."""
    columns = {}
    for metric in METRICS:
        match = [c for c in frame.columns if c == metric or metric in str(c)]
        columns[metric] = match[0] if match else None
    return [
        {m: (_clean(row[c]) if c is not None else None) for m, c in columns.items()}
        for _, row in frame.iterrows()
    ]


def summarize(scores: Sequence[dict]) -> dict[str, float | None]:
    """Mean of each metric, ignoring rows where it could not be computed."""
    summary = {}
    for metric in METRICS:
        values = [s[metric] for s in scores if s.get(metric) is not None]
        summary[metric] = round(sum(values) / len(values), 4) if values else None
    return summary


def _run_ragas(records: list[dict], with_reference: bool, settings: Settings) -> pd.DataFrame:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )

    samples = []
    for record in records:
        fields = dict(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=list(record["contexts"]),
        )
        if with_reference:
            fields["reference"] = record["ground_truth"]
        samples.append(SingleTurnSample(**fields))

    precision = (
        LLMContextPrecisionWithReference() if with_reference
        else LLMContextPrecisionWithoutReference()
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), precision],
        llm=LangchainLLMWrapper(ChatOpenAI(model=settings.eval_model, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model)),
        raise_exceptions=False,
        show_progress=False,
    )
    return result.to_pandas()


def evaluate_records(
    records: Sequence[dict], settings: Settings | None = None, runner=None
) -> list[dict[str, float | None]]:
    """Score many records. Each needs `question`, `answer`, `contexts`; `ground_truth` optional.

    Records without retrieved contexts or without an answer are not scored (all None).
    """
    settings = settings or get_settings()
    if runner is None:
        require_api_key()
        runner = _run_ragas
    empty = {metric: None for metric in METRICS}
    results: list[dict] = [dict(empty) for _ in records]

    groups: dict[bool, list[int]] = {True: [], False: []}
    for index, record in enumerate(records):
        contexts = [c for c in record.get("contexts") or [] if c and c.strip()]
        if not contexts or not (record.get("answer") or "").strip():
            continue
        groups[bool((record.get("ground_truth") or "").strip())].append(index)

    for with_reference, indexes in groups.items():
        if not indexes:
            continue
        frame = runner([records[i] for i in indexes], with_reference, settings)
        for index, scores in zip(indexes, extract_scores(frame)):
            results[index] = scores
    return results


def run_ragas_evaluation(
    query: str,
    response: str,
    retrieved_contexts: Sequence[str],
    ground_truth: str = "",
    settings: Settings | None = None,
) -> dict[str, float | None]:
    """Score a single question/answer pair. Returns {metric: score or None}."""
    record = {
        "question": query,
        "answer": response,
        "contexts": list(retrieved_contexts),
        "ground_truth": ground_truth,
    }
    return evaluate_records([record], settings)[0]
