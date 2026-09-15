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
    UNSOLVED = "UNSOLVED"
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"


ExpandedState = Literal["false", "true", "in progress"]


class TopMove(BaseModel):
    move: str
    visits: int
    prior: float
    avgValue: float
    puct: float | None  # null once a move is solved - see ParallelMCTS::toJSON
    status: SolvedStatus
    isAllowed: bool
    expanded: ExpandedState


class BookEntry(BaseModel):
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
