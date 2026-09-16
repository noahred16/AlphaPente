"""Fragile by nature: runs the real compiled binary at build/<engine>, built
from this same repo (see README for build steps). Uses a temporary RocksDB
(pytest's tmp_path) so this never touches the real book_db."""
import pytest
from rocksdict import Rdict

from api.engine import ENGINE_DIR
from api.tasks.book import evaluate_position, set_allowed_moves

pytestmark = pytest.mark.skipif(
    not (ENGINE_DIR / "pente").exists(),
    reason="build/pente not built; see README for build steps",
)


@pytest.fixture
def db(tmp_path):
    rdict = Rdict(str(tmp_path / "test_book.db"))
    yield rdict
    rdict.close()


def test_evaluate_position_runs_engine_and_persists_result(db):
    # Call the task directly (no broker/worker needed) with few sims.
    result = evaluate_position(["K10", "L9"], target_visits=100, db=db)

    assert result["simulations"] == 100
    assert isinstance(result["bestMove"], str)

    from api.kv_store import get_entry
    from api.zobrist import compute_hash

    entry = get_entry(compute_hash(["K10", "L9"]), db=db)
    assert entry["jobStatus"] == "IDLE"
    assert entry["result"] == result
    assert entry["targetVisits"] == 100  # the requested iteration count, not just what got persisted as `result`


def test_evaluate_position_unregisters_from_the_queue_when_done(db):
    from api.kv_store import get_queued_jobs

    evaluate_position(["K10", "L9"], target_visits=100, db=db)

    assert get_queued_jobs(db=db) == []


def test_evaluate_position_unregisters_from_the_queue_on_failure(db, monkeypatch):
    """The queue registry must not leak an entry forever just because the
    search itself blew up - see evaluate_position's outer finally."""
    from api.kv_store import get_queued_jobs

    def failing_run_search(moves, iterations, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("api.tasks.book.run_search", failing_run_search)

    with pytest.raises(RuntimeError):
        evaluate_position(["K10", "L9"], target_visits=100, db=db)

    assert get_queued_jobs(db=db) == []


def test_evaluate_position_marks_queued_immediately_even_while_waiting_its_turn(db):
    """The actual fix for a real bug: before this write existed, a
    dispatched-but-still-waiting position's persisted jobStatus stayed
    whatever it was *before* being queued (e.g. "IDLE" for a bare stub) for
    the entire time it sat blocked on _search_semaphore under any real
    backlog - long enough for a GET polled during that window to see the
    same stale status and dispatch a *duplicate* evaluate_position for the
    exact same position (see evaluate_position's own docstring, and
    test_router_book.py's test_get_does_not_requeue_a_position_already_dispatched)."""
    import threading

    from api.kv_store import get_entry
    from api.tasks import book as book_tasks
    from api.zobrist import compute_hash

    book_tasks._search_semaphore.acquire()  # simulate another search already in progress
    thread = threading.Thread(
        target=evaluate_position, args=(["K10", "L9"],), kwargs={"target_visits": 100, "db": db}
    )
    try:
        thread.start()
        thread.join(timeout=1)  # never finishes on its own - it's blocked on the semaphore held above
        assert thread.is_alive(), "should still be waiting its turn"

        entry = get_entry(compute_hash(["K10", "L9"]), db=db)
        assert entry["jobStatus"] == "QUEUED"
    finally:
        book_tasks._search_semaphore.release()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_evaluate_position_is_registered_in_the_queue_while_running(db, monkeypatch):
    """Registered from the moment it's dispatched - before it even starts
    searching - so a still-waiting-its-turn job shows up too, not just the
    one actually running (see kv_store.register_queued_job)."""
    import threading
    import time

    from api.kv_store import get_queued_jobs

    started = threading.Event()
    finish = threading.Event()

    def slow_run_search(moves, iterations, *args, **kwargs):
        started.set()
        finish.wait(timeout=5)
        return {
            "simulations": iterations,
            "solvedStatus": "UNSOLVED",
            "rootAvgValue": 0.0,
            "bestMove": None,
            "topMoves": [],
        }

    monkeypatch.setattr("api.tasks.book.run_search", slow_run_search)

    thread = threading.Thread(
        target=evaluate_position, args=(["K10", "L9"],), kwargs={"target_visits": 2_000_000, "db": db}
    )
    thread.start()
    started.wait(timeout=5)

    jobs = get_queued_jobs(db=db)
    assert len(jobs) == 1
    assert jobs[0]["moves"] == ["K10", "L9"]
    assert jobs[0]["targetVisits"] == 2_000_000

    finish.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert get_queued_jobs(db=db) == []


def test_worker_ready_signal_resets_the_queue_registry(db, monkeypatch):
    """The queue registry must start empty on every worker boot - anything
    recorded as in-flight at that moment didn't survive whatever just
    (re)started the process (see api.kv_store.reset_queued_jobs)."""
    import api.tasks.book as book_tasks
    from api.kv_store import get_queued_jobs, register_queued_job

    monkeypatch.setattr(book_tasks, "get_book_db", lambda: db)
    register_queued_job("stale-job", ["K10"], 200_000, db=db)

    book_tasks._reset_queue_registry_on_startup()

    assert get_queued_jobs(db=db) == []


def test_evaluate_position_serializes_concurrent_searches(db, monkeypatch):
    """_search_semaphore exists precisely for this: several evaluate_position
    tasks dispatched at once (e.g. via the worker's --pool=threads) must
    never actually run run_search() at the same time - each search assumes
    it can use the whole machine (NUM_THREADS worker threads, ARENA_SIZE_GB
    memory), so running more than one concurrently oversubscribes both.
    run_search is faked out (with an artificial delay to force real overlap
    between threads) rather than run for real here - the point is proving
    mutual exclusion, not exercising the engine again."""
    import threading
    import time

    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def fake_run_search(moves, iterations, *args, **kwargs):
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.05)
        with lock:
            concurrent -= 1
        return {
            "simulations": iterations,
            "solvedStatus": "UNSOLVED",
            "rootAvgValue": 0.0,
            "bestMove": None,
            "topMoves": [],
        }

    monkeypatch.setattr("api.tasks.book.run_search", fake_run_search)

    threads = [
        threading.Thread(target=evaluate_position, args=([f"A{i + 1}"],), kwargs={"target_visits": 1, "db": db})
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert all(not t.is_alive() for t in threads), "a thread is still stuck"

    assert max_concurrent == 1


def test_set_allowed_moves_persists_and_reports_status(db):
    moves = ["K10", "L9"]
    # Enough sims that the search reliably produces several distinct top
    # moves (this test needs at least 2 to check both the allowed and
    # disallowed cases) - 100 sims can sometimes collapse to just 1.
    result = evaluate_position(moves, target_visits=1000, db=db)
    known_moves = [m["move"] for m in result["topMoves"]]
    assert len(known_moves) >= 2, "test needs a search with multiple candidate moves"
    allowed_move, disallowed_move = known_moves[0], known_moves[1]

    response = set_allowed_moves(moves, [allowed_move], db=db)

    assert response["solvedStatus"] == result["solvedStatus"]
    assert {"move": allowed_move, "isAllowed": True} in response["updatedMoves"]
    assert {"move": disallowed_move, "isAllowed": False} in response["updatedMoves"]

    from api.kv_store import get_entry
    from api.zobrist import compute_hash

    entry = get_entry(compute_hash(moves), db=db)
    assert entry["allowedMoves"] == [allowed_move]
    assert entry["result"] == result  # untouched by the allowed-moves update


def test_set_allowed_moves_reports_moves_set_before_any_evaluation(db):
    """Real bug this reproduces: allowedMoves set on a position with no
    result yet (moves + [nextMove] added ad hoc before its own evaluation
    has run) used to come back with an empty updatedMoves - known_moves was
    built only from result["topMoves"], which doesn't exist yet."""
    moves = ["K10"]

    response = set_allowed_moves(moves, ["L9", "K12"], db=db)

    assert response["solvedStatus"] == "UNSOLVED"
    assert response["updatedMoves"] == [
        {"move": "L9", "isAllowed": True},
        {"move": "K12", "isAllowed": True},
    ]


def test_set_allowed_moves_deletes_a_removed_moves_sole_owned_subtree(db):
    """The trash icon's actual contract: removing a move that was only ever
    reachable through this position deletes its data outright, not just
    hides it - see kv_store.delete_position_if_orphaned."""
    from api.kv_store import get_entry
    from api.zobrist import compute_hash

    moves = ["K10"]
    set_allowed_moves(moves, ["L9"], db=db)
    l9_hash = compute_hash(moves + ["L9"])
    assert get_entry(l9_hash, db=db) is not None  # sanity: it really exists first

    set_allowed_moves(moves, [], db=db)  # trash L9

    assert get_entry(l9_hash, db=db) is None


def test_set_allowed_moves_only_detaches_a_removed_move_with_another_parent(db):
    """A transposition - simulated here via a second, fabricated parentHashes
    edge (see test_kv_store.py for the same check at the kv_store level):
    removing a move from one parent's allowed set must not destroy data
    another position still legitimately relies on - just detach it."""
    from api.kv_store import add_parent_edge, get_entry
    from api.zobrist import compute_hash

    set_allowed_moves(["K10"], ["L9"], db=db)
    l9_hash = compute_hash(["K10", "L9"])
    add_parent_edge(["K10", "L9"], parent_hash="some-other-parent", db=db)  # a second real parent

    set_allowed_moves(["K10"], [], db=db)  # trash L9 from K10's list only

    entry = get_entry(l9_hash, db=db)
    assert entry is not None  # still wanted elsewhere
    assert entry["parentHashes"] == ["some-other-parent"]


def test_evaluate_position_merges_own_top_moves_into_allowed_moves(db):
    """Real bug this reproduces: a move manually added (via set_allowed_moves)
    before its own evaluation finished used to end up as the *only* thing in
    allowedMoves once the search completed - the engine's own top moves
    stayed permanently unapproved (isAllowed: false in the API response)
    unless someone clicked each one by hand. A move the engine itself found
    and ranked should count as allowed by default; opting one out is a
    separate, later PUT that removes it."""
    moves = ["K10"]
    set_allowed_moves(moves, ["L9"], db=db)  # ad hoc, before any evaluation

    result = evaluate_position(moves, target_visits=1000, db=db)
    engine_moves = {m["move"] for m in result["topMoves"]}

    from api.kv_store import get_entry
    from api.zobrist import compute_hash

    entry = get_entry(compute_hash(moves), db=db)
    assert "L9" in entry["allowedMoves"]  # the ad-hoc move survives
    assert engine_moves <= set(entry["allowedMoves"])  # every engine-found move is allowed too


def test_set_allowed_moves_then_evaluate_merges_engine_moves_into_updated_set(db):
    """The other half of the same workflow: once the evaluation this was
    set ahead of actually finishes, a later set_allowed_moves call should
    report both the originally ad-hoc move and whatever the engine itself
    ranked - not lose either side."""
    moves = ["K10"]
    set_allowed_moves(moves, ["L9"], db=db)  # ad hoc, before any evaluation

    result = evaluate_position(moves, target_visits=1000, db=db)
    engine_move = result["topMoves"][0]["move"]

    response = set_allowed_moves(moves, ["L9"], db=db)  # re-set after evaluation finished

    reported_moves = {m["move"] for m in response["updatedMoves"]}
    assert "L9" in reported_moves  # the ad-hoc move is still tracked
    assert engine_move in reported_moves  # the engine's own finding shows up too
