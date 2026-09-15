"""Calls the real compiled pente_native extension (see src/PyBindings.cpp)."""
import pytest

zobrist = pytest.importorskip("api.zobrist", reason="pente_native not built; see README for build steps")


def test_compute_hash_is_deterministic():
    moves = ["K10", "L9", "K12", "M10"]
    assert zobrist.compute_hash(moves) == zobrist.compute_hash(moves)


def test_compute_hash_differs_for_different_positions():
    assert zobrist.compute_hash(["K10", "L9"]) != zobrist.compute_hash(["K10", "L8"])


def test_compute_hash_rejects_illegal_move():
    with pytest.raises(ValueError):
        zobrist.compute_hash(["K10", "K10"])  # same cell twice
