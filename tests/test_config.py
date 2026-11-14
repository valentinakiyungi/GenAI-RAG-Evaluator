import pytest

from src.config import ConfigError, get_settings, has_api_key, require_api_key


def test_defaults_and_overrides(settings, tmp_path):
    assert settings.chunk_size == 120
    assert settings.top_k == 2
    assert settings.chat_model == "gpt-4o"
    assert settings.chroma_dir == tmp_path / "chroma"


@pytest.mark.parametrize(
    "name,value",
    [("CHUNK_SIZE", "abc"), ("CHUNK_SIZE", "0"), ("CHUNK_OVERLAP", "500"),
     ("TOP_K", "0"), ("SEARCH_TYPE", "magic")],
)
def test_invalid_values_are_rejected(settings, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        get_settings()


def test_api_key_detection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "api_key_here")  # placeholder from .env.example
    assert not has_api_key()
    with pytest.raises(ConfigError):
        require_api_key()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert has_api_key()
    require_api_key()
