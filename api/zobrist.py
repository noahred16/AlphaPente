"""Wrapper around the native pente_native extension (pybind11, see
src/PyBindings.cpp) for computing a position's Zobrist hash.

Hashing is cheap and called often (e.g. on every book lookup), unlike
api/engine.py's run_search (an expensive, long-running MCTS job) - so this
calls the real C++ function in-process instead of paying subprocess overhead
per call.
"""
import sys
from pathlib import Path

_BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

import pente_native  # noqa: E402


def compute_hash(moves: list[str]) -> str:
    """Return the hex Zobrist hash of the position reached by `moves` (e.g.
    ["K10", "L9"]) from an empty board. Raises ValueError on an illegal move."""
    return pente_native.compute_hash(moves)


def compute_canonical_hash(moves: list[str]) -> tuple[str, int]:
    """Return (hash, sym) for the position reached by `moves`: a hash that's
    invariant under the 8 board symmetries (so positions that are
    rotations/reflections of each other share the same value), and which of
    those 8 symmetries (0-7) maps this specific physical position to that
    canonical form - see apply_symmetry. Raises ValueError on an illegal move."""
    return pente_native.compute_canonical_hash(moves)


def apply_symmetry(move: str, sym: int, inverse: bool = False) -> str:
    """Apply symmetry `sym` (or its inverse) to a move label, returning the
    transformed label - e.g. to re-express a move found via one orientation's
    search in a different orientation's coordinate frame. Raises ValueError
    if `move` isn't a well-formed move label."""
    return pente_native.apply_symmetry(move, sym, inverse)
