from fastapi import APIRouter, HTTPException, Query

from api.schemas.book import (
    AllowedMovesRequest,
    AllowedMovesResponse,
    BookEntry,
    EvaluateRequest,
    EvaluateResponse,
)

router = APIRouter(prefix="/pente/book", tags=["book"])


@router.get("", response_model=BookEntry)
def get_book_entry(moves: list[str] = Query(default=[])) -> BookEntry:
    """Return move evaluation and opening book state for a position."""
    # TODO: fetch node state from book_db (RocksDict), propagate solved status
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("", response_model=EvaluateResponse)
def queue_evaluation(body: EvaluateRequest) -> EvaluateResponse:
    """Queue an MCTS evaluation job for a position."""
    # TODO: enqueue book.evaluate, persist job status in book_db (RocksDict)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.put("/allowed-moves", response_model=AllowedMovesResponse)
def update_allowed_moves(body: AllowedMovesRequest) -> AllowedMovesResponse:
    """Update the allowed-move subset and re-evaluate solved status up the parent DAG."""
    # TODO: update book_db (RocksDict), recompute solved status propagation
    raise HTTPException(status_code=501, detail="Not implemented")
