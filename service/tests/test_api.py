from fastapi.testclient import TestClient

from tamu_ai.api import app

client = TestClient(app)


def test_health():
    assert client.get("/ai/v1/health").json() == {"status": "ok"}


def test_search_rejects_bad_requests():
    assert client.post("/ai/v1/search", json={"q": ""}).status_code == 422
    assert client.post("/ai/v1/search", json={"q": "maps", "mode": "magic"}).status_code == 422
