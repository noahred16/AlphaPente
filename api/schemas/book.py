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
    childHash: str
    visits: int
    prior: float
    avgValue: float
    puct: float
    status: SolvedStatus
    isAllowed: bool
    expanded: ExpandedState


class BookEntry(BaseModel):
    uniq_hash: str
    jobStatus: JobStatus
    totalVisits: int
    solvedStatus: SolvedStatus
    bestValue: float
    bestMove: str
    date_started: datetime
    topMoves: list[TopMove]


class EvaluateRequest(BaseModel):
    targetVisits: int | None = None


class EvaluateResponse(BaseModel):
    job_id: str
    uniq_hash: str
    jobStatus: JobStatus


class AllowedMovesRequest(BaseModel):
    allowedMoves: list[str]


class UpdatedMove(BaseModel):
    move: str
    isAllowed: bool


class AllowedMovesResponse(BaseModel):
    uniq_hash: str
    solvedStatus: SolvedStatus
    updatedMoves: list[UpdatedMove]
