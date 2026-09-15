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
