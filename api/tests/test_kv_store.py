"""Uses a temporary RocksDB (pytest's tmp_path) so tests never touch the
real book_db (see api/config.py's book_db_path)."""
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
