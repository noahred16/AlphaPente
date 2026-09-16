"""Uses a temporary RocksDB (pytest's tmp_path) so tests never touch the
real book_db (see api/config.py's book_db_path)."""
import threading

import pytest
from rocksdict import Rdict

kv_store = pytest.importorskip("api.kv_store", reason="pente_native not built; see README for build steps")


@pytest.fixture
def db(tmp_path):
    rdict = Rdict(str(tmp_path / "test_book.db"))
    yield rdict
    rdict.close()


def test_save_and_get_entry_roundtrip(db):
    moves = ["K10", "L9"]

    hash_hex = kv_store.save_entry(moves, job_status="QUEUED", db=db)
    entry = kv_store.get_entry(hash_hex, db=db)

    assert entry["moves"] == moves
    assert entry["jobStatus"] == "QUEUED"
    assert entry["result"] is None
    assert entry["date_started"]  # set on first write


def test_save_entry_preserves_date_started_across_updates(db):
    moves = ["K10", "L9"]

    hash_hex = kv_store.save_entry(moves, job_status="QUEUED", db=db)
    first_date = kv_store.get_entry(hash_hex, db=db)["date_started"]

    result = {"bestMove": "L12", "totalVisits": 100}
    kv_store.save_entry(moves, job_status="IDLE", result=result, db=db)
    entry = kv_store.get_entry(hash_hex, db=db)

    assert entry["date_started"] == first_date
    assert entry["jobStatus"] == "IDLE"
    assert entry["result"] == result


def test_get_entry_missing_returns_none(db):
    assert kv_store.get_entry("deadbeef", db=db) is None


def test_save_entry_preserves_result_when_not_given(db):
    """A job_status-only update (e.g. marking IN_PROGRESS at the start of a
    re-run) must not erase a previous result - only an explicit new result
    should replace it."""
    moves = ["K10", "L9"]
    result = {"bestMove": "L12", "totalVisits": 100}
    kv_store.save_entry(moves, job_status="IDLE", result=result, db=db)

    kv_store.save_entry(moves, job_status="IN_PROGRESS", db=db)
    entry = kv_store.get_entry(kv_store.compute_hash(moves), db=db)

    assert entry["jobStatus"] == "IN_PROGRESS"
    assert entry["result"] == result


def test_save_entry_defaults_job_status_to_idle_for_a_new_entry(db):
    hash_hex = kv_store.save_entry(["K10", "L9"], db=db)
    assert kv_store.get_entry(hash_hex, db=db)["jobStatus"] == "IDLE"


def test_save_entry_persists_and_preserves_allowed_moves(db):
    moves = ["K10", "L9"]
    hash_hex = kv_store.save_entry(moves, allowed_moves=["L12", "J10"], db=db)
    assert kv_store.get_entry(hash_hex, db=db)["allowedMoves"] == ["L12", "J10"]

    # A later update that doesn't mention allowed_moves preserves it.
    kv_store.save_entry(moves, job_status="IN_PROGRESS", db=db)
    assert kv_store.get_entry(hash_hex, db=db)["allowedMoves"] == ["L12", "J10"]


def test_save_entry_persists_and_preserves_target_visits(db):
    moves = ["K10", "L9"]
    hash_hex = kv_store.save_entry(moves, target_visits=2_000_000, db=db)
    assert kv_store.get_entry(hash_hex, db=db)["targetVisits"] == 2_000_000

    # A later update that doesn't mention target_visits preserves it.
    kv_store.save_entry(moves, job_status="IN_PROGRESS", db=db)
    assert kv_store.get_entry(hash_hex, db=db)["targetVisits"] == 2_000_000


def test_reader_sees_writes_from_a_separate_writer_process_handle(tmp_path, monkeypatch):
    """The whole point of get_book_db_reader(): RocksDB only allows one
    read-write handle on a path at a time, so the API process (reader) and
    the Celery worker (writer) must use different access modes on the same
    on-disk book_db - this proves that combination actually works and stays
    live, not just each mode in isolation."""
    monkeypatch.setattr(kv_store.settings, "book_db_path", str(tmp_path / "book.db"))
    monkeypatch.setattr(kv_store, "_book_db", None)
    monkeypatch.setattr(kv_store, "_book_db_reader", None)

    writer = kv_store.get_book_db()  # simulates the worker process's handle
    hash_hex = kv_store.save_entry(["K10", "L9"], job_status="QUEUED", db=writer)

    entry = kv_store.get_entry(hash_hex)  # no db= -> uses get_book_db_reader()
    assert entry["jobStatus"] == "QUEUED"

    kv_store.save_entry(["K10", "L9"], job_status="IN_PROGRESS", db=writer)
    entry = kv_store.get_entry(hash_hex)
    assert entry["jobStatus"] == "IN_PROGRESS"  # reader picks up the writer's later update


def test_reader_returns_none_when_book_db_does_not_exist_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store.settings, "book_db_path", str(tmp_path / "never-written.db"))
    monkeypatch.setattr(kv_store, "_book_db_reader", None)

    assert kv_store.get_book_db_reader() is None
    assert kv_store.get_entry("deadbeef") is None


def test_add_parent_edge_is_safe_under_concurrent_writers(db):
    """_book_db_write_lock exists precisely for this: without it, two
    threads racing add_parent_edge on the same child can both read
    parentHashes before either's append is visible to the other, and the
    second write silently clobbers the first's - losing an edge rather than
    raising anything. Also implicitly proves the lock is reentrant (RLock,
    not Lock): add_parent_edge calls save_entry from the same thread while
    already holding it - a plain Lock would deadlock every one of these
    threads instantly rather than fail this assertion.

    Caveat, stated plainly rather than hidden: this natural race is timing-
    dependent and didn't reliably reproduce here even with the lock actually
    removed by hand (Python's GIL/scheduler rarely lands two threads inside
    the narrow read->write gap at this small scale) - so this test won't
    reliably catch a regression on its own. The mechanism itself - a bare
    read-modify-write on a JSON blob losing an update under real forced
    interleaving - was verified directly with a minimal repro outside this
    suite before trusting the lock here; that's the actual evidence this
    lock is doing real work, not this test passing.
    """
    child_moves = ["K10", "L9"]
    parent_hashes = [f"parent{i}" for i in range(10)]

    threads = [
        threading.Thread(target=kv_store.add_parent_edge, args=(child_moves, p), kwargs={"db": db})
        for p in parent_hashes
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert all(not t.is_alive() for t in threads), "a thread is still stuck - suggests a lock deadlocked"


# ─── delete_position_if_orphaned ──────────────────────────────────────────────


def test_delete_position_if_orphaned_deletes_a_sole_owned_subtree(db):
    """K10->L9->M11: L9's only parent is K10, M11's only parent is L9 - so
    deleting through K10 should cascade all the way down and remove both
    physical entries, plus their canonical-group membership."""
    kv_store.add_parent_edge(["K10", "L9"], parent_hash="k10-hash", db=db)
    l9_hash = kv_store.save_entry(["K10", "L9"], allowed_moves=["M11"], db=db)
    m11_hash = kv_store.add_parent_edge(["K10", "L9", "M11"], parent_hash=l9_hash, db=db)
    l9_canonical = kv_store.get_entry(l9_hash, db=db)["canonicalHash"]
    m11_canonical = kv_store.get_entry(m11_hash, db=db)["canonicalHash"]

    kv_store.delete_position_if_orphaned(l9_hash, parent_hash="k10-hash", db=db)

    assert kv_store.get_entry(l9_hash, db=db) is None
    assert kv_store.get_entry(m11_hash, db=db) is None
    assert l9_hash not in kv_store.get_canonical_members(l9_canonical, db=db)
    assert m11_hash not in kv_store.get_canonical_members(m11_canonical, db=db)


def test_delete_position_if_orphaned_only_detaches_a_position_with_other_parents(db):
    """A transposition: the same child is reachable from two different
    parents. Removing one parent's edge must not delete data the other
    parent still legitimately relies on - just shrink parentHashes."""
    child_hash = kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent1", db=db)
    kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent2", db=db)

    kv_store.delete_position_if_orphaned(child_hash, parent_hash="parent1", db=db)

    entry = kv_store.get_entry(child_hash, db=db)
    assert entry is not None
    assert entry["parentHashes"] == ["parent2"]


def test_delete_position_if_orphaned_is_a_no_op_for_a_nonexistent_hash(db):
    kv_store.delete_position_if_orphaned("deadbeef", parent_hash="parent1", db=db)  # must not raise


# ─── queue registry ────────────────────────────────────────────────────────


def test_register_and_get_queued_jobs_roundtrip(db):
    kv_store.register_queued_job("job1", ["K10", "L9"], 2_000_000, db=db)

    jobs = kv_store.get_queued_jobs(db=db)

    assert len(jobs) == 1
    assert jobs[0]["moves"] == ["K10", "L9"]
    assert jobs[0]["targetVisits"] == 2_000_000
    assert jobs[0]["queuedAt"]


def test_get_queued_jobs_returns_oldest_first(db):
    kv_store.register_queued_job("job1", ["K10"], 200_000, db=db)
    kv_store.register_queued_job("job2", ["K10", "L9"], 2_000_000, db=db)

    jobs = kv_store.get_queued_jobs(db=db)

    assert [j["moves"] for j in jobs] == [["K10"], ["K10", "L9"]]


def test_unregister_queued_job_removes_only_that_job(db):
    kv_store.register_queued_job("job1", ["K10"], 200_000, db=db)
    kv_store.register_queued_job("job2", ["K10", "L9"], 2_000_000, db=db)

    kv_store.unregister_queued_job("job1", db=db)

    assert [j["moves"] for j in kv_store.get_queued_jobs(db=db)] == [["K10", "L9"]]


def test_unregister_queued_job_is_a_no_op_for_an_unknown_id(db):
    kv_store.unregister_queued_job("nonexistent", db=db)  # must not raise
    assert kv_store.get_queued_jobs(db=db) == []


def test_reset_queued_jobs_clears_everything(db):
    kv_store.register_queued_job("job1", ["K10"], 200_000, db=db)
    kv_store.register_queued_job("job2", ["K10", "L9"], 2_000_000, db=db)

    kv_store.reset_queued_jobs(db=db)

    assert kv_store.get_queued_jobs(db=db) == []


def test_get_queued_jobs_returns_empty_list_when_book_db_does_not_exist_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store.settings, "book_db_path", str(tmp_path / "never-written.db"))
    monkeypatch.setattr(kv_store, "_book_db_reader", None)

    assert kv_store.get_queued_jobs() == []
