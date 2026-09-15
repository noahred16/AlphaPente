from rocksdict import Rdict

from api.celery_app import celery_app
from api.engine import run_search
from api.kv_store import save_entry
from api.schemas.book import JobStatus

# Previously run by hand with a huge iteration count and left to search until
# the tree arena filled up. Used as the default when no targetVisits is given.
FULL_SEARCH_ITERATIONS = 900_000_000


@celery_app.task(name="book.evaluate")
def evaluate_position(moves: list[str], target_visits: int | None = None, db: Rdict | None = None) -> dict:
    """Run MCTS evaluation on a position and persist the result in book_db.

    `db` is a test-only escape hatch (defaults to the real book_db) - it
    isn't JSON-serializable, so never pass it through .delay()/apply_async();
    only call this directly (as tests do) with an explicit db.
    """
    save_entry(moves, job_status=JobStatus.IN_PROGRESS.value, db=db)

    iterations = target_visits or FULL_SEARCH_ITERATIONS
    try:
        result = run_search(moves, iterations)
    except Exception:
        save_entry(moves, job_status=JobStatus.FAILED.value, db=db)
        raise

    save_entry(moves, job_status=JobStatus.IDLE.value, result=result, db=db)
    return result
