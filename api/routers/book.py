from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.kv_store import get_entry
from api.schemas.book import (
    AllowedMovesRequest,
    AllowedMovesResponse,
    BookEntry,
    EvaluateRequest,
    EvaluateResponse,
    JobStatus,
    SolvedStatus,
    TopMove,
)
from api.tasks.book import evaluate_position
from api.zobrist import compute_hash

router = APIRouter(prefix="/pente/book", tags=["book"])


def _to_book_entry(entry: dict) -> BookEntry:
    """Map a stored book_db entry ({moves, jobStatus, date_started, result})
    onto the API's BookEntry shape. `result` is the engine's own JSON output
    (see GameUtils::runSearchAndReportJSON) - None if no search has completed
    yet (freshly queued/in-progress, or never evaluated at all)."""
    result = entry.get("result")
    if result is None:
        return BookEntry(
            jobStatus=entry["jobStatus"],
            totalVisits=0,
            solvedStatus=SolvedStatus.UNSOLVED,
            bestValue=0.0,
            bestMove=None,
            date_started=entry["date_started"],
            topMoves=[],
        )

    return BookEntry(
        jobStatus=entry["jobStatus"],
        totalVisits=result["totalVisits"],
        solvedStatus=result["solvedStatus"],
        bestValue=result["rootAvgValue"],
        bestMove=result["bestMove"],
        date_started=entry["date_started"],
        topMoves=[
            TopMove(
                move=m["move"],
                visits=m["visits"],
                prior=m["prior"],
                avgValue=m["avgValue"],
                puct=m["puct"],
                status=m["status"],
                # TODO: reflect PUT /allowed-moves once implemented, instead
                # of defaulting every move to allowed/unexpanded.
                isAllowed=True,
                expanded="false",
            )
            for m in result["topMoves"]
        ],
    )


@router.get("", response_model=BookEntry)
def get_book_entry(moves: list[str] = Query(default=[])) -> BookEntry:
    """Return move evaluation and opening book state for a position. If this
    position has never been seen before, queues an evaluation for it (same
    as POST) and returns a freshly-QUEUED entry rather than 404ing."""
    try:
        hash_hex = compute_hash(moves)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    entry = get_entry(hash_hex)
    if entry is None:
        evaluate_position.delay(moves)
        entry = {
            "jobStatus": JobStatus.QUEUED.value,
            "date_started": datetime.now(timezone.utc).isoformat(),
            "result": None,
        }

    return _to_book_entry(entry)


@router.post("", response_model=EvaluateResponse)
def queue_evaluation(body: EvaluateRequest) -> EvaluateResponse:
    """Queue an MCTS evaluation job for a position.

    Doesn't write to book_db itself - only the Celery worker (evaluate_position)
    owns writes to it, since RocksDB allows only one read-write handle on a
    given path at a time; see api/kv_store.py. compute_hash still validates
    the moves up front (a pure call, no book_db access) so illegal input
    fails fast with a 400 instead of surfacing later as a failed task.
    """
    try:
        compute_hash(body.moves)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    task = evaluate_position.delay(body.moves, target_visits=body.targetVisits)
    return EvaluateResponse(job_id=task.id, jobStatus=JobStatus.QUEUED)


@router.put("/allowed-moves", response_model=AllowedMovesResponse)
def update_allowed_moves(body: AllowedMovesRequest) -> AllowedMovesResponse:
    """Update the allowed-move subset and re-evaluate solved status up the parent DAG."""
    # TODO: this is a write, so - like POST - it can't touch book_db directly
    # from this process (see queue_evaluation's docstring); route it through
    # a Celery task owned by the worker, then recompute solved status propagation
    raise HTTPException(status_code=501, detail="Not implemented")
