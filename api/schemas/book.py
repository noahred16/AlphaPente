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


class SearchLevel(str, Enum):
    # How hard to search a position - the only lever exposed for that now;
    # each maps to a hardcoded iteration count (see
    # api/tasks/book.py's SEARCH_LEVEL_ITERATIONS).
    VERY_FAST = "VERY_FAST"
    FAST = "FAST"
    MEDIUM = "MEDIUM"
    DEEP = "DEEP"


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
    # The targetVisits the child's own last evaluation was run at (see
    # BookEntry.targetVisits) - None if it's never been evaluated at all.
    # Lets the table show each child's own search level (Fast/Medium/Deep)
    # and offer re-running one deeper without having to drill into it first.
    childTargetVisits: int | None = None
    # Whether `move`'s own child position currently has a real search job
    # running - independent of `expanded` above, which deliberately keeps
    # reporting "true" off a still-good older result while a same-level
    # re-run (or a deepen) is in flight rather than masking it with "in
    # progress" (see _to_book_entry.expanded_state's docstring). Lets the
    # frontend still know a just-requested re-run has actually been picked
    # up by the worker, so it can stop assuming so locally - see
    # docs/js/book.js's deepeningMoves.
    childInProgress: bool = False
    # The avgValue the engine found for `move`'s own child's best reply - one
    # search deeper than avgValue itself - already in this position's own
    # to-move player's perspective (two plies down, so no sign flip is
    # needed here, unlike avgValue - see ParallelMCTS::backpropagate's
    # per-ply flip). None if the child's own search hasn't run yet, or found
    # no legal replies at all.
    childBestMoveValue: float | None = None


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
    # The targetVisits this position's own last evaluation was run at - None
    # if it's never been evaluated (still QUEUED/IN_PROGRESS with no prior
    # result, or a bare stub). See TopMove.childTargetVisits for the same
    # thing one ply down, per candidate move.
    targetVisits: int | None = None


class QueuedJob(BaseModel):
    # A single evaluate_position call dispatched but not yet finished -
    # including the one actually running right now, since there's no
    # reliable way to tell which registry entry that is (see
    # kv_store.register_queued_job) - the frontend's ETA math treats the
    # oldest one as "running now" instead.
    moves: list[str]
    targetVisits: int
    queuedAt: datetime
    # Rough wall-clock estimate for how long this job itself takes once it
    # starts - None for DEEP (arena-bound, not time-bound - see
    # api/tasks/book.py's SEARCH_LEVEL_SECONDS). The frontend chains these
    # together, in order, into each job's own estimated start/end and an
    # overall "queue empty at" time.
    estimatedSeconds: int | None = None


class QueueResponse(BaseModel):
    count: int
    jobs: list[QueuedJob]


class EvaluateRequest(BaseModel):
    moves: list[MoveStr]
    level: SearchLevel = SearchLevel.MEDIUM


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
