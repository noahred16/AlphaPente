from rocksdict import Rdict

from api.celery_app import celery_app
from api.engine import run_search
from api.kv_store import add_parent_edge, get_book_db, get_entry, propagate_book_status, save_entry
from api.schemas.book import JobStatus

# Used as the default when no targetVisits is given. Was 900_000_000 (run by
# hand, left to search until the tree arena filled up) - dropped to a ~30s
# run for now while debugging; bump back up once the API itself is settled.
FULL_SEARCH_ITERATIONS = 5_000_000


@celery_app.task(name="book.evaluate")
def evaluate_position(moves: list[str], target_visits: int | None = None, db: Rdict | None = None) -> dict:
    """Run MCTS evaluation on a position and persist the result in book_db.

    `db` is a test-only escape hatch (defaults to the real book_db) - it
    isn't JSON-serializable, so never pass it through .delay()/apply_async();
    only call this directly (as tests do) with an explicit db.
    """
    db = db if db is not None else get_book_db()
    save_entry(moves, job_status=JobStatus.IN_PROGRESS.value, db=db)

    iterations = target_visits or FULL_SEARCH_ITERATIONS
    try:
        result = run_search(moves, iterations)
    except Exception:
        save_entry(moves, job_status=JobStatus.FAILED.value, db=db)
        raise

    hash_hex = save_entry(moves, job_status=JobStatus.IDLE.value, result=result, db=db)

    # Record this position as the parent of every candidate reply it found,
    # then recompute (and, if changed, propagate up) its own book-level
    # solved status - see kv_store.propagate_book_status.
    for top_move in result["topMoves"]:
        add_parent_edge(moves + [top_move["move"]], parent_hash=hash_hex, db=db)
    propagate_book_status(hash_hex, db=db)

    return result


@celery_app.task(name="book.set_allowed_moves")
def set_allowed_moves(moves: list[str], allowed_moves: list[str], db: Rdict | None = None) -> dict:
    """Persist the allowed-move subset for a position (doesn't touch its
    jobStatus/result - see save_entry's merge semantics), recompute its
    book-level solved status against the new set and propagate any change up
    the parent DAG, then report back the result."""
    db = db if db is not None else get_book_db()
    hash_hex = save_entry(moves, allowed_moves=allowed_moves, db=db)
    propagate_book_status(hash_hex, db=db)
    entry = get_entry(hash_hex, db=db)

    result = entry.get("result")
    known_moves = [m["move"] for m in result["topMoves"]] if result else []

    return {
        "solvedStatus": entry["bookSolvedStatus"],
        "updatedMoves": [{"move": m, "isAllowed": m in allowed_moves} for m in known_moves],
    }
