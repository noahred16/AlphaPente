"""Router-level tests for /pente/book. Never touches a real Celery broker
(patches evaluate_position.delay) or the real book_db (GET's reads are
redirected to a temporary RocksDB; POST doesn't touch book_db at all - see
queue_evaluation's docstring)."""
import pytest
from fastapi.testclient import TestClient
from rocksdict import Rdict

from api.main import app
from api.schemas.book import SearchLevel
from api.tasks.book import SEARCH_LEVEL_ITERATIONS, evaluate_position, set_allowed_moves

client = TestClient(app)


class _FakeAsyncResult:
    id = "fake-task-id"

    def __init__(self, value=None):
        self._value = value

    def get(self, timeout=None):
        return self._value


def test_post_queues_job(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.post("/pente/book", json={"moves": ["K10", "L9"], "level": "FAST"})

    assert response.status_code == 200
    assert response.json() == {"job_id": "fake-task-id", "jobStatus": "QUEUED"}
    assert calls == [((["K10", "L9"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST]})]


def test_post_queues_job_at_very_fast_level(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.post("/pente/book", json={"moves": ["K10", "L9"], "level": "VERY_FAST"})

    assert response.status_code == 200
    assert calls == [((["K10", "L9"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.VERY_FAST]})]


def test_post_defaults_to_medium_level(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.post("/pente/book", json={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    assert calls == [((["K10", "L9"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.MEDIUM]})]


def test_post_rejects_illegal_move():
    response = client.post("/pente/book", json={"moves": ["K10", "K10"]})

    assert response.status_code == 400


def test_post_rejects_malformed_move():
    # "P0" is well-formed enough to reach compute_hash but segfaults it (a
    # real gap in PenteGame::makeMove - see test_propagation.py's module
    # docstring); the MoveStr pattern on the schema rejects it before that.
    response = client.post("/pente/book", json={"moves": ["P0"]})

    assert response.status_code == 422


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
    assert body["moves"] == ["K10", "L9"]  # so a freshly-queued position can still be built on immediately
    assert body["jobStatus"] == "QUEUED"
    assert body["totalVisits"] == 0
    assert body["bestMove"] is None
    assert body["topMoves"] == []
    assert calls == [((["K10", "L9"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.MEDIUM]})]


def test_get_does_not_requeue_a_position_already_dispatched(reader_db, monkeypatch):
    """Real bug this reproduces: evaluate_position used to leave book_db's
    persisted jobStatus untouched (e.g. still "IDLE") for as long as a
    dispatched job sat waiting behind _search_semaphore - the "QUEUED"
    needs_evaluation checks for was only ever a throwaway dict built for that
    one response, never actually written to book_db. Every GET polled during
    that whole wait saw the same stale status and dispatched *another*
    evaluate_position for the exact same position - see
    evaluate_position's own docstring for the full story."""
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    client.get("/pente/book", params={"moves": ["K10", "L9"]})
    assert len(calls) == 1

    # Simulate evaluate_position's real first action (see its docstring) -
    # the persisted state a position sits in for as long as it's still
    # waiting its turn, well before any result exists.
    from api.kv_store import save_entry

    save_entry(["K10", "L9"], job_status="QUEUED", db=reader_db)

    client.get("/pente/book", params={"moves": ["K10", "L9"]})
    assert len(calls) == 1  # not dispatched again


def test_get_queues_evaluation_at_the_requested_level(reader_db, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    response = client.get("/pente/book", params={"moves": ["K10", "L9"], "level": "FAST"})

    assert response.status_code == 200
    assert calls == [((["K10", "L9"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST]})]


def test_can_queue_a_followup_move_while_parent_is_still_unresolved(reader_db, monkeypatch):
    """The actual workflow this is for: click into a position that's still
    QUEUED/unresolved (GET), build the next move purely from its echoed
    `moves`, and POST that follow-up right away - without waiting for the
    parent's own evaluation to finish."""
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    parent_response = client.get("/pente/book", params={"moves": ["K10"]})
    assert parent_response.status_code == 200
    parent_body = parent_response.json()
    assert parent_body["jobStatus"] == "QUEUED"  # still unresolved - no result, no topMoves to pick from

    followup_moves = parent_body["moves"] + ["L9"]
    followup_response = client.post("/pente/book", json={"moves": followup_moves})

    assert followup_response.status_code == 200
    assert followup_response.json()["jobStatus"] == "QUEUED"
    medium_visits = SEARCH_LEVEL_ITERATIONS[SearchLevel.MEDIUM]
    assert calls == [
        ((["K10"],), {"target_visits": medium_visits}),  # the parent, queued by GET
        ((followup_moves,), {"target_visits": medium_visits}),  # the follow-up, queued by POST
    ]


def test_get_queues_a_bare_entry_left_behind_by_put_allowed_moves(reader_db, monkeypatch):
    """Real bug this reproduces: PUT /allowed-moves (or add_parent_edge)
    creates a bare entry - jobStatus IDLE, no result - for a position that's
    never actually been queued for evaluation. Before this fix, GET's
    auto-queue only checked `entry is None`, which is already false once
    that bare entry exists - so the position could get stuck showing IDLE
    with no result forever, un-fixable by GET alone."""
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())
    # Run the real set_allowed_moves logic (against reader_db) rather than a
    # stub, so PUT actually leaves the bare entry behind, same as it does
    # for real - only the Celery dispatch itself is faked.
    monkeypatch.setattr(
        set_allowed_moves, "delay", lambda *a, **kw: _FakeAsyncResult(set_allowed_moves(*a, db=reader_db, **kw))
    )

    response = client.put("/pente/book/allowed-moves", json={"moves": ["K10"], "allowedMoves": ["L9"]})
    assert response.status_code == 200
    assert calls == []  # PUT itself never queues an evaluation

    response = client.get("/pente/book", params={"moves": ["K10"]})

    assert response.status_code == 200
    body = response.json()
    assert body["jobStatus"] == "QUEUED"
    assert calls == [((["K10"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.MEDIUM]})]
    # The actual point of allowedMoves showing up in topMoves even with no
    # result yet: "L9" is visible and queueable right now, not after a
    # search finishes.
    assert [m["move"] for m in body["topMoves"]] == ["L9"]
    assert body["topMoves"][0]["isAllowed"] is True


def test_get_retries_a_failed_entry(reader_db, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult())

    from api.kv_store import save_entry

    save_entry(["K10"], job_status="FAILED", db=reader_db)

    response = client.get("/pente/book", params={"moves": ["K10"]})

    assert response.status_code == 200
    assert response.json()["jobStatus"] == "QUEUED"
    assert calls == [((["K10"],), {"target_visits": SEARCH_LEVEL_ITERATIONS[SearchLevel.MEDIUM]})]


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
            "symmetricTo": None,
            "childMoveCount": 0,
            "childTargetVisits": None,
            "childInProgress": False,
            "childBestMoveValue": None,
        }
    ]


def test_get_surfaces_allowed_move_outside_top_moves(reader_db, monkeypatch):
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
    save_entry(["K10", "L9"], job_status="IDLE", result=result, allowed_moves=["G10", "A1"], db=reader_db)

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    body = response.json()
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
            "symmetricTo": None,
            "childMoveCount": 0,
            "childTargetVisits": None,
            "childInProgress": False,
            "childBestMoveValue": None,
        },
        {
            "move": "A1",
            "visits": 0,
            "prior": 0.0,
            "avgValue": None,
            "puct": None,
            "status": "UNSOLVED",
            "isAllowed": True,
            "expanded": "false",
            "symmetricTo": None,
            "childMoveCount": 0,
            "childTargetVisits": None,
            "childInProgress": False,
            "childBestMoveValue": None,
        },
    ]


def test_get_reflects_child_expanded_state(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
            {"move": "H11", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
            {"move": "J12", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, db=reader_db)
    # G10 has its own completed search - IN_PROGRESS is fine here since it's
    # already overridden by the result being present.
    save_entry(["K10", "L9", "G10"], job_status="IDLE", result={"totalVisits": 5}, db=reader_db)
    # H11 is queued/running but hasn't produced a result yet.
    save_entry(["K10", "L9", "H11"], job_status="IN_PROGRESS", db=reader_db)
    # J12 is left untouched - only a bare add_parent_edge-style stub, never queued.

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    expanded_by_move = {m["move"]: m["expanded"] for m in response.json()["topMoves"]}
    assert expanded_by_move == {"G10": "true", "H11": "in progress", "J12": "false"}


def test_get_reports_child_in_progress_independent_of_a_lingering_result(reader_db, monkeypatch):
    """A re-run/deepen of an already-expanded move leaves its old result in
    place (see save_entry's merge semantics) while jobStatus goes
    IN_PROGRESS - `expanded` deliberately still reads "true" off that old
    result (see test_get_reflects_child_expanded_state), but
    `childInProgress` should still see through to the real, running job."""
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
            {"move": "H11", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, db=reader_db)
    # G10 is being re-run right now, but still has its old result.
    save_entry(["K10", "L9", "G10"], job_status="IN_PROGRESS", result={"totalVisits": 5}, db=reader_db)
    # H11 has never been touched at all - genuinely not in progress either.

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    body = response.json()
    in_progress_by_move = {m["move"]: m["childInProgress"] for m in body["topMoves"]}
    assert in_progress_by_move == {"G10": True, "H11": False}
    # And `expanded` is unaffected - still masked by the lingering result.
    assert {m["move"]: m["expanded"] for m in body["topMoves"]} == {"G10": "true", "H11": "false"}


def test_get_prefers_a_childs_own_book_status_over_a_stale_engine_snapshot(reader_db, monkeypatch):
    """Real bug this reproduces: a move ("J11" in production) can fall out of
    the parent's own frozen topMoves snapshot (reported UNSOLVED, or missing
    entirely and shown via the allowedMoves stub) while its own child
    position has since been independently proven SOLVED_LOSS - a decisive
    fact the parent's stale one-ply read has no way to know about on its
    own. The child's current bookSolvedStatus should win."""
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            # G10: this search's own one-ply read still says UNSOLVED, and
            # the child has never been independently searched either.
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, allowed_moves=["G10", "H11"], db=reader_db)
    # H11 fell outside this search's own topMoves (only reachable via the
    # allowedMoves stub) but has since been deeply, independently proven.
    save_entry(
        ["K10", "L9", "H11"],
        job_status="IDLE",
        result={"solvedStatus": "SOLVED_LOSS", "topMoves": []},
        db=reader_db,
    )
    from api.kv_store import propagate_book_status
    from api.zobrist import compute_hash

    propagate_book_status(compute_hash(["K10", "L9", "H11"]), db=reader_db)

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    status_by_move = {m["move"]: m["status"] for m in response.json()["topMoves"]}
    assert status_by_move == {"G10": "UNSOLVED", "H11": "SOLVED_LOSS"}


def test_get_reports_child_move_count(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
            {"move": "H11", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, db=reader_db)
    # G10's own child has 3 moves already in its allowed list.
    save_entry(["K10", "L9", "G10"], job_status="IDLE", allowed_moves=["A1", "A2", "A3"], db=reader_db)
    # H11 is left untouched - only a bare add_parent_edge-style stub, never queued.

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    count_by_move = {m["move"]: m["childMoveCount"] for m in response.json()["topMoves"]}
    assert count_by_move == {"G10": 3, "H11": 0}


def test_get_reports_child_best_move_value(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
            {"move": "H11", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9"], job_status="IDLE", result=result, db=reader_db)
    # G10's own child has its own independent search - its own bestMove
    # ("A2") is what child_best_move_value should read the avgValue of, not
    # its other, non-best candidate ("A1").
    child_result = {
        "totalVisits": 500,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": 0.1,
        "bestMove": "A2",
        "topMoves": [
            {"move": "A1", "visits": 100, "prior": 0.2, "avgValue": -0.5, "puct": 0.1, "status": "UNSOLVED"},
            {"move": "A2", "visits": 400, "prior": 0.3, "avgValue": 0.42, "puct": 0.2, "status": "UNSOLVED"},
        ],
    }
    save_entry(["K10", "L9", "G10"], job_status="IDLE", result=child_result, db=reader_db)
    # H11 is left untouched - never independently searched.

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    value_by_move = {m["move"]: m["childBestMoveValue"] for m in response.json()["topMoves"]}
    assert value_by_move == {"G10": 0.42, "H11": None}


def test_get_reports_target_visits_and_child_target_visits(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    result = {
        "totalVisits": 1000,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": -0.04,
        "bestMove": "G10",
        "topMoves": [
            {"move": "G10", "visits": 786, "prior": 0.0417, "avgValue": 0.0687, "puct": 0.0729, "status": "UNSOLVED"},
            {"move": "H11", "visits": 500, "prior": 0.0417, "avgValue": 0.01, "puct": 0.02, "status": "UNSOLVED"},
        ],
    }
    save_entry(
        ["K10", "L9"], job_status="IDLE", result=result, target_visits=SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST],
        db=reader_db,
    )
    # G10's own child was searched deeper; H11 is left untouched (never evaluated).
    save_entry(
        ["K10", "L9", "G10"], job_status="IDLE", target_visits=SEARCH_LEVEL_ITERATIONS[SearchLevel.DEEP],
        db=reader_db,
    )

    response = client.get("/pente/book", params={"moves": ["K10", "L9"]})

    assert response.status_code == 200
    body = response.json()
    assert body["targetVisits"] == SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST]
    child_target_visits_by_move = {m["move"]: m["childTargetVisits"] for m in body["topMoves"]}
    assert child_target_visits_by_move == {"G10": SEARCH_LEVEL_ITERATIONS[SearchLevel.DEEP], "H11": None}


def test_get_flags_symmetric_duplicates_within_top_moves(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import save_entry

    # From an empty board, A1 and A19 are genuine symmetric twins of each
    # other (corner-to-corner reflection); B2 is a separate symmetry class -
    # found empirically the same way as the propagation tests' twin pairs.
    result = {
        "totalVisits": 100,
        "solvedStatus": "UNSOLVED",
        "rootAvgValue": 0.0,
        "bestMove": "A1",
        "topMoves": [
            {"move": "A1", "visits": 50, "prior": 0.1, "avgValue": 0.0, "puct": 0.1, "status": "UNSOLVED"},
            {"move": "A19", "visits": 50, "prior": 0.1, "avgValue": 0.0, "puct": 0.1, "status": "UNSOLVED"},
            {"move": "B2", "visits": 10, "prior": 0.1, "avgValue": 0.0, "puct": 0.1, "status": "UNSOLVED"},
        ],
    }
    save_entry([], job_status="IDLE", result=result, db=reader_db)

    response = client.get("/pente/book", params={"moves": []})

    assert response.status_code == 200
    symmetric_to_by_move = {m["move"]: m["symmetricTo"] for m in response.json()["topMoves"]}
    assert symmetric_to_by_move == {"A1": None, "A19": "A1", "B2": None}


def test_get_translates_a_solved_twins_result_instead_of_queueing(reader_db, monkeypatch):
    monkeypatch.setattr(evaluate_position, "delay", lambda *a, **kw: pytest.fail("shouldn't queue a job"))

    from api.kv_store import propagate_book_status, save_entry
    from api.zobrist import apply_symmetry, compute_canonical_hash

    # ["A1", "B2"] and ["A19", "B18"] are genuine board-symmetric twins with
    # completely different move histories (see test_propagation.py) - only
    # the first is ever actually evaluated here. Derive the expected
    # translated label the same way the implementation does, rather than
    # guessing coordinates by hand.
    twin_moves = ["A1", "B2"]
    query_moves = ["A19", "B18"]
    candidate_move = "C3"
    _, twin_sym = compute_canonical_hash(twin_moves)
    _, query_sym = compute_canonical_hash(query_moves)
    expected_move = apply_symmetry(apply_symmetry(candidate_move, twin_sym), query_sym, inverse=True)

    twin_result = {
        "totalVisits": 100,
        "solvedStatus": "SOLVED_WIN",
        "rootAvgValue": 1.0,
        "bestMove": candidate_move,
        "topMoves": [
            {"move": candidate_move, "visits": 100, "prior": 0.5, "avgValue": 1.0, "puct": None, "status": "SOLVED_WIN"},
        ],
    }
    twin_hash = save_entry(twin_moves, job_status="IDLE", result=twin_result, db=reader_db)
    propagate_book_status(twin_hash, db=reader_db)

    response = client.get("/pente/book", params={"moves": query_moves})

    assert response.status_code == 200
    body = response.json()
    assert body["solvedStatus"] == "SOLVED_WIN"  # shared via the canonical group
    assert body["bestMove"] == expected_move
    assert [m["move"] for m in body["topMoves"]] == [expected_move]


def test_get_rejects_illegal_move(reader_db):
    response = client.get("/pente/book", params={"moves": ["K10", "K10"]})

    assert response.status_code == 400


def test_get_rejects_malformed_move(reader_db):
    response = client.get("/pente/book", params={"moves": ["P0"]})

    assert response.status_code == 422


def test_get_queue_reports_pending_jobs_oldest_first(reader_db):
    from api.kv_store import register_queued_job

    register_queued_job("job1", ["K10"], SEARCH_LEVEL_ITERATIONS[SearchLevel.VERY_FAST], db=reader_db)
    register_queued_job("job2", ["K10", "L9"], SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST], db=reader_db)

    response = client.get("/pente/book/queue")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [j["moves"] for j in body["jobs"]] == [["K10"], ["K10", "L9"]]
    assert [j["targetVisits"] for j in body["jobs"]] == [
        SEARCH_LEVEL_ITERATIONS[SearchLevel.VERY_FAST],
        SEARCH_LEVEL_ITERATIONS[SearchLevel.FAST],
    ]
    assert [j["estimatedSeconds"] for j in body["jobs"]] == [5, 30]


def test_get_queue_reports_no_estimate_for_a_deep_job(reader_db):
    from api.kv_store import register_queued_job

    register_queued_job("job1", ["K10"], SEARCH_LEVEL_ITERATIONS[SearchLevel.DEEP], db=reader_db)

    response = client.get("/pente/book/queue")

    assert response.json()["jobs"][0]["estimatedSeconds"] is None


def test_get_queue_is_empty_when_nothing_is_pending(reader_db):
    response = client.get("/pente/book/queue")

    assert response.status_code == 200
    assert response.json() == {"count": 0, "jobs": []}


def test_put_allowed_moves_updates_and_returns_status(monkeypatch):
    payload = {"solvedStatus": "UNSOLVED", "updatedMoves": [{"move": "L12", "isAllowed": True}]}
    calls = []
    monkeypatch.setattr(
        set_allowed_moves, "delay", lambda *a, **kw: calls.append((a, kw)) or _FakeAsyncResult(payload)
    )

    response = client.put("/pente/book/allowed-moves", json={"moves": ["K10", "L9"], "allowedMoves": ["L12"]})

    assert response.status_code == 200
    assert response.json() == payload
    assert calls == [((["K10", "L9"], ["L12"]), {})]


def test_put_allowed_moves_rejects_illegal_move():
    response = client.put("/pente/book/allowed-moves", json={"moves": ["K10", "K10"], "allowedMoves": []})

    assert response.status_code == 400


def test_put_allowed_moves_rejects_malformed_move():
    response = client.put("/pente/book/allowed-moves", json={"moves": ["K10"], "allowedMoves": ["P0"]})

    assert response.status_code == 422
