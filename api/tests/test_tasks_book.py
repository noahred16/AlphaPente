"""Fragile by nature: runs the real compiled binary at build/<engine>, built
from this same repo (see README for build steps). Uses a temporary RocksDB
(pytest's tmp_path) so this never touches the real book_db."""
import pytest
from rocksdict import Rdict

from api.engine import ENGINE_DIR
from api.tasks.book import evaluate_position

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
