import pandas as pd
import pytest

from src.evaluate import METRICS
from src.history import clear_history, load_history, log_run
from src.ingest import ingest_file
from src.pipeline import answer_question, evaluate_result, parse_test_set, run_batch


def fake_runner(records, with_reference, _settings):
    return pd.DataFrame({metric: [0.8] * len(records) for metric in METRICS})


@pytest.fixture
def indexed(settings, vectorstore, handbook):
    ingest_file(handbook, settings, vectorstore)
    return vectorstore


def test_answer_question_end_to_end(settings, indexed, llm):
    result = answer_question(
        "How many days of annual leave?", settings=settings, vectorstore=indexed, llm=llm
    )
    assert result.answer == llm.reply
    assert len(result.contexts) == settings.top_k
    assert "28 days" in llm.calls[0][1][1]  # retrieved text reached the prompt
    assert result.scores is None and result.latency_s >= 0

    evaluate_result(result, settings, fake_runner)
    assert result.scores == {metric: 0.8 for metric in METRICS}
    assert result.to_dict()["contexts"][0]["source"] == "handbook.txt"


def test_run_batch_scores_logs_and_survives_bad_rows(settings, indexed, llm):
    rows = [
        {"question": "How many days of leave?", "ground_truth": "28 days"},
        {"question": "", "ground_truth": ""},  # fails: empty question
        {"question": "How long to submit expenses?"},
    ]
    seen = []
    table = run_batch(
        rows, settings=settings, vectorstore=indexed, llm=llm, runner=fake_runner,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert list(table["faithfulness"].fillna(-1)) == [0.8, -1, 0.8]
    assert table.loc[1, "error"].startswith("ValueError")
    assert table.loc[0, "sources"] == "handbook.txt"

    history = load_history(settings.history_file)
    assert len(history) == 2  # the failed row is not logged
    assert history["faithfulness"].tolist() == [0.8, 0.8]


def test_parse_test_set():
    frame = pd.DataFrame({" Question ": ["a", None, " "], "Ground_Truth": ["x", "y", None]})
    assert parse_test_set(frame) == [{"question": "a", "ground_truth": "x"}]
    assert parse_test_set(pd.DataFrame({"question": ["a"]})) == [
        {"question": "a", "ground_truth": ""}
    ]
    with pytest.raises(ValueError):
        parse_test_set(pd.DataFrame({"prompt": ["a"]}))
    with pytest.raises(ValueError):
        parse_test_set(pd.DataFrame({"question": [None]}))


def test_history_roundtrip(tmp_path):
    path = tmp_path / "logs" / "history.jsonl"
    assert load_history(path).empty
    log_run(path, {"question": "q", "answer": "a", "contexts": [{"source": "b.pdf"}, {"source": "a.pdf"}],
                   "scores": {"faithfulness": 0.5}, "latency_s": 1.2})
    log_run(path, {"question": "q2", "answer": "a2", "contexts": [], "scores": None})
    with path.open("a") as handle:
        handle.write("not json\n")
    history = load_history(path)
    assert len(history) == 2
    assert history.loc[0, "sources"] == ["a.pdf", "b.pdf"]
    assert history.loc[0, "faithfulness"] == 0.5
    assert pd.isna(history.loc[1, "faithfulness"])
    clear_history(path)
    assert load_history(path).empty
