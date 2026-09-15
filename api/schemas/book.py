from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel


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
    puct: float
    status: SolvedStatus
    isAllowed: bool
    expanded: ExpandedState


class BookEntry(BaseModel):
    jobStatus: JobStatus
    totalVisits: int
    solvedStatus: SolvedStatus
    bestValue: float
    bestMove: str
    date_started: datetime
    topMoves: list[TopMove]


class EvaluateRequest(BaseModel):
    moves: list[str]
    targetVisits: int | None = None


class EvaluateResponse(BaseModel):
    job_id: str
    jobStatus: JobStatus


class AllowedMovesRequest(BaseModel):
    moves: list[str]
    allowedMoves: list[str]


class UpdatedMove(BaseModel):
    move: str
    isAllowed: bool


class AllowedMovesResponse(BaseModel):
    solvedStatus: SolvedStatus
    updatedMoves: list[UpdatedMove]
