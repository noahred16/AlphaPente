from fastapi import APIRouter, HTTPException

from api.schemas.book import (
    AllowedMovesRequest,
    AllowedMovesResponse,
    BookEntry,
    EvaluateRequest,
    EvaluateResponse,
)

router = APIRouter(prefix="/pente/book", tags=["book"])


@router.get("/{uniq_hash}", response_model=BookEntry)
def get_book_entry(uniq_hash: str) -> BookEntry:
    """Return move evaluation and opening book state for a position hash."""
    # TODO: fetch node state from book_db (RocksDict), propagate solved status
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/{uniq_hash}", response_model=EvaluateResponse)
def queue_evaluation(uniq_hash: str, body: EvaluateRequest | None = None) -> EvaluateResponse:
    """Queue an MCTS evaluation job for a position hash."""
    # TODO: enqueue Celery task, persist job status in book_db (RocksDict)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.put("/{uniq_hash}/allowed-moves", response_model=AllowedMovesResponse)
def update_allowed_moves(uniq_hash: str, body: AllowedMovesRequest) -> AllowedMovesResponse:
    """Update the allowed-move subset and re-evaluate solved status up the parent DAG."""
    # TODO: update book_db (RocksDict), recompute solved status propagation
    raise HTTPException(status_code=501, detail="Not implemented")
