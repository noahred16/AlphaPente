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
    MoveStr,
    SolvedStatus,
    TopMove,
)
from api.tasks.book import evaluate_position, set_allowed_moves
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

    # None (never set via PUT /allowed-moves) means "nothing restricted yet" - allow everything.
    allowed_moves = entry.get("allowedMoves")

    return BookEntry(
        jobStatus=entry["jobStatus"],
        totalVisits=result["totalVisits"],
        # bookSolvedStatus (propagated across separately-evaluated children,
        # restricted to allowedMoves - see kv_store.propagate_book_status),
        # not result["solvedStatus"] (that one only reflects what this one
        # bounded search found on its own).
        solvedStatus=entry.get("bookSolvedStatus", "UNSOLVED"),
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
                isAllowed=allowed_moves is None or m["move"] in allowed_moves,
                # TODO: reflect whether this child has its own book_db entry,
                # instead of always reporting unexpanded.
                expanded="false",
            )
            for m in result["topMoves"]
        ],
    )


@router.get("", response_model=BookEntry)
def get_book_entry(moves: list[MoveStr] = Query(default=[])) -> BookEntry:
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
    """Persist the allowed-move subset for a position.

    Like POST, this is a write and can't touch book_db directly from this
    process (see queue_evaluation's docstring) - it runs via a Celery task.
    Unlike POST, it waits synchronously for that task rather than returning
    a job_id: the response needs the task's result (solvedStatus,
    updatedMoves), and persisting an allowed-move subset plus propagating
    any solved-status change is fast, unlike an actual MCTS search.
    """
    try:
        compute_hash(body.moves)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    result = set_allowed_moves.delay(body.moves, body.allowedMoves).get(timeout=10)
    return AllowedMovesResponse(**result)
