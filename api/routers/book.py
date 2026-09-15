from fastapi import APIRouter, HTTPException, Query

from api.schemas.book import (
    AllowedMovesRequest,
    AllowedMovesResponse,
    BookEntry,
    EvaluateRequest,
    EvaluateResponse,
    JobStatus,
)
from api.tasks.book import evaluate_position
from api.zobrist import compute_hash

router = APIRouter(prefix="/pente/book", tags=["book"])


@router.get("", response_model=BookEntry)
def get_book_entry(moves: list[str] = Query(default=[])) -> BookEntry:
    """Return move evaluation and opening book state for a position."""
    # TODO: fetch node state from book_db (RocksDict), propagate solved status
    raise HTTPException(status_code=501, detail="Not implemented")


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
