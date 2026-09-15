from api.celery_app import celery_app
from api.engine import run_search

# Previously run by hand with a huge iteration count and left to search until
# the tree arena filled up. Used as the default when no targetVisits is given.
FULL_SEARCH_ITERATIONS = 900_000_000


@celery_app.task(name="book.evaluate")
def evaluate_position(moves: list[str], target_visits: int | None = None) -> dict:
    """Run MCTS evaluation on a position and report back the engine's result."""
    iterations = target_visits or FULL_SEARCH_ITERATIONS
    result = run_search(moves, iterations)
    # TODO: persist result in book_db (RocksDict)
    return result
