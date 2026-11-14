import pandas as pd
import pytest

from src.config import ConfigError
from src.evaluate import METRICS, evaluate_records, extract_scores, run_ragas_evaluation, summarize


def test_extract_scores_handles_ragas_column_names_and_nan():
    frame = pd.DataFrame(
        {
            "user_input": ["q1", "q2"],
            "faithfulness": [1.0, float("nan")],
            "answer_relevancy": [0.91234, 0.5],
            "llm_context_precision_without_reference": [0.75, 1.0],
        }
    )
    assert extract_scores(frame) == [
        {"faithfulness": 1.0, "answer_relevancy": 0.9123, "context_precision": 0.75},
        {"faithfulness": None, "answer_relevancy": 0.5, "context_precision": 1.0},
    ]


def test_summarize_ignores_missing_values():
    scores = [
        {"faithfulness": 1.0, "answer_relevancy": None, "context_precision": 0.5},
        {"faithfulness": 0.5, "answer_relevancy": None, "context_precision": None},
    ]
    assert summarize(scores) == {
        "faithfulness": 0.75, "answer_relevancy": None, "context_precision": 0.5,
    }


def test_records_are_grouped_by_reference_and_unscorable_ones_skipped(settings):
    calls = []

    def runner(records, with_reference, _settings):
        calls.append((with_reference, [r["question"] for r in records]))
        value = 0.9 if with_reference else 0.4
        return pd.DataFrame({metric: [value] * len(records) for metric in METRICS})

    records = [
        {"question": "a", "answer": "x", "contexts": ["c"], "ground_truth": ""},
        {"question": "b", "answer": "x", "contexts": ["c"], "ground_truth": "truth"},
        {"question": "c", "answer": "x", "contexts": [], "ground_truth": "truth"},
        {"question": "d", "answer": "", "contexts": ["c"]},
        {"question": "e", "answer": "x", "contexts": ["c"]},
    ]
    scores = evaluate_records(records, settings, runner)
    assert sorted(calls) == [(False, ["a", "e"]), (True, ["b"])]
    assert [s["faithfulness"] for s in scores] == [0.4, 0.9, None, None, 0.4]


def test_missing_api_key_gives_clear_error(settings):
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        run_ragas_evaluation("q", "a", ["context"], settings=settings)
