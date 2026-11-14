import pytest

from src.config import ConfigError
from src.generator import NO_CONTEXT_ANSWER, build_messages, generate_answer, stream_answer
from src.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk("Employees receive 28 days of leave.", "handbook.pdf", page=2, score=0.9),
    RetrievedChunk("Remote work is allowed three days a week.", "handbook.pdf", page=5),
]


def test_prompt_contains_numbered_context_and_question():
    system, human = build_messages("  How much leave?  ", CHUNKS)
    assert system[0] == "system" and "ONLY" in system[1]
    assert "[1] (source: handbook.pdf, p. 2)" in human[1]
    assert "[2] (source: handbook.pdf, p. 5)" in human[1]
    assert human[1].endswith("Question: How much leave?")


def test_generate_and_stream(llm):
    assert generate_answer("How much leave?", CHUNKS, llm=llm) == llm.reply
    assert "".join(stream_answer("How much leave?", CHUNKS, llm=llm)).strip() == llm.reply
    assert len(llm.calls) == 2


def test_no_context_skips_the_model(llm):
    assert generate_answer("anything", [], llm=llm) == NO_CONTEXT_ANSWER
    assert list(stream_answer("anything", [], llm=llm)) == [NO_CONTEXT_ANSWER]
    assert llm.calls == []


def test_missing_api_key_gives_clear_error(settings):
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        generate_answer("How much leave?", CHUNKS, settings)
