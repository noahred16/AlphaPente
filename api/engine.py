"""Run the compiled game engine as a subprocess and parse its JSON search output.

See GameUtils::runSearchAndReportJSON (src/GameUtils.cpp) for the JSON shape.
"""
import json
import subprocess
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent / "build"


class EngineError(RuntimeError):
    """Raised when the engine binary exits non-zero or produces bad output."""


def run_search(moves: list[str], iterations: int, engine: str = "pente", engine_dir: Path = ENGINE_DIR) -> dict:
    """Run `engine` on the position reached by `moves` (e.g. ["K10", "L9"])
    for `iterations` MCTS simulations, and return its parsed JSON search
    result (bestMove, topMoves, solvedStatus, etc.)."""
    position = " ".join(moves)
    cmd = [f"./{engine}", position, str(iterations), "-n", "-j"]
    result = subprocess.run(cmd, cwd=engine_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise EngineError(f"{engine} exited {result.returncode}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise EngineError(f"{engine} produced invalid JSON: {e}") from e
