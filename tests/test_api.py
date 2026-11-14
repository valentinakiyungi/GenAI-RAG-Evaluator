import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)


def test_health(settings):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "openai_api_key_configured": False}


def test_missing_api_key_returns_503(settings):
    response = client.post("/query", json={"question": "How much leave?"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_validation(settings):
    assert client.post("/query", json={"question": ""}).status_code == 422
    bad = client.post("/query", json={"question": "x", "search_type": "magic"})
    assert bad.status_code == 400
