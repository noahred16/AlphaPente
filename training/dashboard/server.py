"""Read-only viewer for self-play games already sitting in a run's buffer.pt.

Shells out to the `replay` binary (apps/Replay.cpp) exactly the way
api/engine.py shells out to `pente` — buffer.pt is a libtorch tensor archive
and this process has no torch installed, so decoding it only happens in C++.

No caching: a full-buffer decode is ~1.3s even at the 62,500-position cap
(see apps/TrainCommon.hpp's BUFFER_SIZE), so each request just re-runs it.
"""
import json
import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPLAY_BIN = os.environ.get("REPLAY_BIN", "/opt/pente-build/replay")
RUNS_DIR = os.environ.get("RUNS_DIR", "/data/runs")
RUN_NAME = os.environ.get("RUN_NAME", "pente_v1_medium")
TOP_K = 5

app = FastAPI()


def buffer_path() -> str:
    return os.path.join(RUNS_DIR, RUN_NAME, "buffer.pt")


def load_games() -> list[dict]:
    path = buffer_path()
    if not os.path.exists(path):
        raise HTTPException(404, f"No buffer.pt for run '{RUN_NAME}' at {path}")

    result = subprocess.run(
        [REPLAY_BIN, "-b", path, "-n", "1000000", "-j", "-k", str(TOP_K)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise HTTPException(500, f"replay failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)["games"]
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(500, f"replay produced unparseable output: {e}")


# z's sign convention (see SelfPlay.cpp's backfill comment): a ply's own z is
# +1 if the player who moved INTO that ply — i.e. that ply's own `move.mover`
# — wins. The last ply in a game is the position right before the actual
# game-ending move, so its z/mover tells us the real outcome directly.
def derive_winner(plies: list[dict]) -> str | None:
    if len(plies) < 2:
        return None
    last = plies[-1]
    move = last.get("move")
    if not move:
        return None
    z = last["z"]
    if z > 0.5:
        return move["mover"]
    if z < -0.5:
        return "white" if move["mover"] == "black" else "black"
    return "draw"


@app.get("/api/games")
def list_games(page: int = 1, page_size: int = 20):
    games = load_games()
    start = (page - 1) * page_size
    page_games = games[start:start + page_size]
    return {
        "run": RUN_NAME,
        "page": page,
        "page_size": page_size,
        "total_games": len(games),
        "games": [
            {
                "index": g["index"],
                "partial": g["partial"],
                "plies": len(g["plies"]),
                "winner": derive_winner(g["plies"]),
                "final_black_captures": g["plies"][-1]["black_captures"],
                "final_white_captures": g["plies"][-1]["white_captures"],
            }
            for g in page_games
        ],
    }


@app.get("/api/games/{index}")
def get_game(index: int):
    games = load_games()
    match = next((g for g in games if g["index"] == index), None)
    if match is None:
        raise HTTPException(404, f"No game at index {index} (0 = most recent)")
    return {**match, "winner": derive_winner(match["plies"])}


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))
