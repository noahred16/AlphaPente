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


def test_save_and_get_result_roundtrip(db):
    moves = ["K10", "L9"]
    result = {"bestMove": "L12", "totalVisits": 100}

    hash_hex = kv_store.save_result(moves, result, db=db)
    stored = kv_store.get_result(hash_hex, db=db)

    assert stored == {"moves": moves, "result": result}


def test_get_result_missing_returns_none(db):
    assert kv_store.get_result("deadbeef", db=db) is None
