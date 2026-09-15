"""Fragile by nature: runs the real compiled binary at build/<engine>, built
from this same repo (see README for build steps)."""
import pytest

from api.engine import ENGINE_DIR, run_search

pytestmark = pytest.mark.skipif(
    not (ENGINE_DIR / "pente").exists(),
    reason="build/pente not built; see README for build steps",
)


def test_run_search_returns_engine_json():
    # Few sims: just confirm the subprocess call and JSON parsing work.
    result = run_search(["K10", "L9"], 100)

    assert result["simulations"] == 100
    assert isinstance(result["bestMove"], str)
    assert isinstance(result["topMoves"], list)
    assert result["topMoves"], "expected at least one candidate move"
