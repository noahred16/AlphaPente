"""Fragile by nature: runs the real compiled binary at build/<engine>, built
from this same repo (see README for build steps)."""
import pytest

from api.engine import ENGINE_DIR
from api.tasks.book import evaluate_position

pytestmark = pytest.mark.skipif(
    not (ENGINE_DIR / "pente").exists(),
    reason="build/pente not built; see README for build steps",
)


def test_evaluate_position_runs_engine():
    # Call the task directly (no broker/worker needed) with few sims.
    result = evaluate_position(["K10", "L9"], target_visits=100)

    assert result["simulations"] == 100
    assert isinstance(result["bestMove"], str)
