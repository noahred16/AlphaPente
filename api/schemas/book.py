from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

# Column A-T skipping I (matches GameUtils::parseMove's "skip I" convention),
# row 1-19 (19x19 board - the only size the API ever computes hashes for).
# Rejects a malformed label like "P0" or "I5" here, at the request-validation
# layer, before it ever reaches compute_hash: PenteGame::makeMove/parseMove
# don't validate their input and segfault on some malformed labels (e.g. a
# row of "0") rather than raising - see api/tests/test_propagation.py's
# module docstring. A validly-formatted but illegal move (e.g. an
# already-occupied cell) still isn't caught here - that's still handled by
# the ValueError -> 400 already in place around compute_hash calls.
MoveStr = Annotated[str, StringConstraints(pattern=r"^[A-HJ-T](?:[1-9]|1[0-9])$")]


class JobStatus(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"


class SolvedStatus(str, Enum):
    # Matches ParallelMCTS::solvedStatusName's actual JSON strings exactly
    # (see ParallelMCTS::toJSON) - the engine never emits bare "WIN"/"LOSS"/
    # "DRAW", only these. Getting this wrong isn't a cosmetic mismatch: it's
    # a Pydantic ValidationError (500) the instant a real solved position's
    # own result or a topMove's status reaches this schema.
    UNSOLVED = "UNSOLVED"
    SOLVED_WIN = "SOLVED_WIN"
    SOLVED_LOSS = "SOLVED_LOSS"
    SOLVED_DRAW = "SOLVED_DRAW"


ExpandedState = Literal["false", "true", "in progress"]


class TopMove(BaseModel):
    move: str
    visits: int
    prior: float
    avgValue: float | None  # null only for a manually-added move with no engine data yet - see _to_book_entry
    puct: float | None  # null once a move is solved - see ParallelMCTS::toJSON
    status: SolvedStatus
    isAllowed: bool
    expanded: ExpandedState
    # Set when an earlier move in this same topMoves list leads to a
    # board-symmetric twin of this move's own child position (see
    # kv_store.compute_canonical_hash) - i.e. two "different" candidates that
    # are actually the same choice mirrored. Points at that earlier move;
    # None if this is the first (or only) move seen for its symmetry class.
    symmetricTo: str | None = None
    # How many moves are already in the allowed-move list for the position
    # this move leads to - a quick sense, from the parent's own table, of how
    # much book already exists past this move without having to drill in.
    # 0 if the child has never been touched at all.
    childMoveCount: int = 0


class BookEntry(BaseModel):
    # Echoed back so the caller always has the exact move sequence this
    # entry represents, regardless of whether a search has completed yet -
    # e.g. to queue a follow-up move (POST moves + [nextMove]) right after
    # clicking into a still-QUEUED/IN_PROGRESS position, without having to
    # independently track the move history client-side.
    moves: list[str]
    jobStatus: JobStatus
    totalVisits: int
    solvedStatus: SolvedStatus
    bestValue: float
    bestMove: str | None  # null: no completed search yet (freshly queued, or root has no children)
    date_started: datetime
    topMoves: list[TopMove]


class EvaluateRequest(BaseModel):
    moves: list[MoveStr]
    targetVisits: int | None = None


class EvaluateResponse(BaseModel):
    job_id: str
    jobStatus: JobStatus


class AllowedMovesRequest(BaseModel):
    moves: list[MoveStr]
    allowedMoves: list[MoveStr]


class UpdatedMove(BaseModel):
    move: str
    isAllowed: bool


class AllowedMovesResponse(BaseModel):
    solvedStatus: SolvedStatus
    updatedMoves: list[UpdatedMove]
