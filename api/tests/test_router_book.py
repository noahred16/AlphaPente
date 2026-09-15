"""Router-level tests for /pente/book. Never touches a real Celery broker
(patches evaluate_position.delay) or the real book_db (GET's reads are
redirected to a temporary RocksDB; POST doesn't touch book_db at all - see
queue_evaluation's docstring)."""
import pytest
from fastapi.testclient import TestClient
from rocksdict import Rdict

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


@pytest.fixture
def reader_db(tmp_path, monkeypatch):
    """Redirects GET's book_db reads (api.kv_store.get_entry's default) to a
    temporary RocksDB instead of the real one."""
    db = Rdict(str(tmp_path / "test_book.db"))
    monkeypatch.setattr("api.kv_store.get_book_db_reader", lambda: db)
    yield db
    db.close()


def test_get_queues_evaluation_when_position_unseen(reader_db, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    body = response.json()
    assert body["jobStatus"] == "QUEUED"
    assert body["totalVisits"] == 0
    assert body["bestMove"] is None
    assert body["topMoves"] == []
    assert calls == [((["K10", "L9"],), {})]


def test_get_returns_existing_entry_with_result(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, db=reader_db)

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    body = response.json()
    assert body["jobStatus"] == "IDLE"
    assert body["totalVisits"] == 1000
    assert body["bestMove"] == "G10"
    assert body["bestValue"] == -0.04
    assert body["topMoves"] == [
        {
            "move": "G10",
            "visits": 786,
            "prior": 0.0417,
            "avgValue": 0.0687,
            "puct": 0.0729,
            "status": "UNSOLVED",
            "isAllowed": True,
            "expanded": "false",
        }
    ]


def test_get_rejects_illegal_move(reader_db):
    response = client.get("/pente/book", params={"moves": ["K10", "K10"]})

    assert response.status_code == 400
