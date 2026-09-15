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
