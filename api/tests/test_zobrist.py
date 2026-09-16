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


def test_compute_canonical_hash_matches_across_symmetric_positions():
    # K10 is the board center; L9/L11 reflect across the horizontal line
    # through it, so these are genuine symmetric twins.
    hash1, _ = zobrist.compute_canonical_hash(["K10", "L9"])
    hash2, _ = zobrist.compute_canonical_hash(["K10", "L11"])
    assert hash1 == hash2


def test_compute_canonical_hash_differs_for_non_symmetric_positions():
    hash1, _ = zobrist.compute_canonical_hash(["K10", "L9"])
    hash2, _ = zobrist.compute_canonical_hash(["K10", "M8"])
    assert hash1 != hash2


def test_apply_symmetry_roundtrips_via_its_inverse():
    for sym in range(8):
        transformed = zobrist.apply_symmetry("L9", sym)
        assert zobrist.apply_symmetry(transformed, sym, inverse=True) == "L9"


def test_apply_symmetry_rejects_malformed_move():
    with pytest.raises(ValueError):
        zobrist.apply_symmetry("P0", 0)
