"""Router-level tests for POST /pente/book. Never touches a real Celery
broker (patches evaluate_position.delay) or any book_db (the router doesn't
write to it at all - see queue_evaluation's docstring)."""
from fastapi.testclient import TestClient

from api.main import app
from api.tasks.book import evaluate_position

client = TestClient(app)


class _FakeAsyncResult:
    id = "fake-task-id"


def test_post_queues_job(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.post("/pente/book", json={"moves": ["K10", "L9"], "targetVisits": 1000})

    assert response.status_code == 200
    assert response.json() == {"job_id": "fake-task-id", "jobStatus": "QUEUED"}
    assert calls == [((["K10", "L9"],), {"target_visits": 1000})]


def test_post_rejects_illegal_move():
    response = client.post("/pente/book", json={"moves": ["K10", "K10"]})

    assert response.status_code == 400
