import threading

from rocksdict import Rdict

from api.celery_app import celery_app
from api.engine import run_search
from api.kv_store import (
    add_parent_edge,
    delete_position_if_orphaned,
    get_book_db,
    get_entry,
    propagate_book_status,
    save_entry,
)
from api.schemas.book import JobStatus
from api.zobrist import compute_hash

# Used as the default when no targetVisits is given. Was 900_000_000 (run by
# hand, left to search until the tree arena filled up) - dropped to a ~30s
# run for now while debugging; bump back up once the API itself is settled.
FULL_SEARCH_ITERATIONS = 5_000_000

# Each pente search assumes it can use the whole machine - NUM_THREADS worker
# threads, ARENA_SIZE_GB memory (see docker-compose.yml) - so running several
# concurrently massively oversubscribes both. Confirmed in practice: with the
# worker's --pool=threads (needed so fast tasks like set_allowed_moves aren't
# stuck behind a slow evaluate_position - see that service's comment), 8
# concurrent searches each claimed ~29GB of arena, causing system-wide
# thrashing severe enough that set_allowed_moves timed out waiting for a CPU
# slice. This gates only the actual run_search() call below - set_allowed_moves
# and everything else stay unaffected, free to run concurrently on the
# pool's other threads while a search is in progress.
#
# A backlogged evaluate_position just blocks on acquire() below until it's
# its turn - it does NOT retry/reschedule itself. That was tried first (to
# avoid a blocked task occupying a thread-pool slot) but it's real polling:
# every backlogged task wakes up, fails, and re-enqueues itself every couple
# seconds for as long as the current search runs, which is both noisy (a
# burst of retry log lines every cycle, for every backlogged task at once)
# and pointless busywork. A thread blocked on a semaphore costs nothing (no
# CPU, just a sleeping OS thread), so the actual fix is to make sure the
# pool has more threads (see docker-compose.yml's --concurrency) than any
# realistic backlog - anything beyond that genuinely waits, untouched, in
# Redis, which is the real "queue" the backlog stacks up in.
_search_semaphore = threading.Semaphore(1)


@celery_app.task(name="book.evaluate")
def evaluate_position(moves: list[str], target_visits: int | None = None, db: Rdict | None = None) -> dict:
    """Run MCTS evaluation on a position and persist the result in book_db.
    Only one evaluate_position actually searches at a time - see
    _search_semaphore above - regardless of how many are dispatched
    concurrently; jobStatus stays whatever it already was (e.g. QUEUED)
    until this one actually starts, not IN_PROGRESS while merely waiting
    for a turn.

    `db` is a test-only escape hatch (defaults to the real book_db) - it
    isn't JSON-serializable, so never pass it through .delay()/apply_async();
    only call this directly (as tests do) with an explicit db.
    """
    _search_semaphore.acquire()

    db = db if db is not None else get_book_db()
    save_entry(moves, job_status=JobStatus.IN_PROGRESS.value, db=db)

    iterations = target_visits or FULL_SEARCH_ITERATIONS
    try:
        result = run_search(moves, iterations)
    except Exception:
        save_entry(moves, job_status=JobStatus.FAILED.value, db=db)
        raise
    finally:
        _search_semaphore.release()

    # A move the engine itself found and ranked is, by definition, in scope
    # for this position's proof - union it into allowedMoves rather than
    # leaving it to require a separate manual approval. Preserves anything
    # already there (e.g. added by hand before this search finished, or kept
    # from a previous search's allowedMoves) - this only ever adds, it never
    # drops a move someone already removed via a later, smaller PUT.
    existing_allowed = (get_entry(compute_hash(moves), db=db) or {}).get("allowedMoves") or []
    engine_moves = [m["move"] for m in result["topMoves"]]
    merged_allowed = list(dict.fromkeys([*existing_allowed, *engine_moves]))

    hash_hex = save_entry(moves, job_status=JobStatus.IDLE.value, result=result, allowed_moves=merged_allowed, db=db)

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
    the parent DAG, then report back the result.

    A move newly added here gets a real parentHashes edge too (add_parent_edge),
    just like a move the engine's own search discovers - not just for
    evaluate_position's benefit, but so a *later* removal can trust
    parentHashes enough to know whether this was the move's only real parent
    (see delete_position_if_orphaned) rather than always assuming it was
    never wanted anywhere else. A move dropped from the set is exactly that
    removal: its subtree is deleted outright if this was its sole parent,
    or just detached (parentHashes shrinks by one) if some other position
    still legitimately has it in its own allowedMoves.
    """
    db = db if db is not None else get_book_db()
    hash_hex = compute_hash(moves)
    previously_allowed = (get_entry(hash_hex, db=db) or {}).get("allowedMoves") or []

    hash_hex = save_entry(moves, allowed_moves=allowed_moves, db=db)

    for added_move in allowed_moves:
        if added_move not in previously_allowed:
            add_parent_edge(moves + [added_move], parent_hash=hash_hex, db=db)
    for removed_move in previously_allowed:
        if removed_move not in allowed_moves:
            delete_position_if_orphaned(compute_hash(moves + [removed_move]), parent_hash=hash_hex, db=db)

    propagate_book_status(hash_hex, db=db)
    entry = get_entry(hash_hex, db=db)

    # Union of the engine's own ranked candidates (once they exist) and the
    # allowedMoves just set - not just the former, or a position with no
    # result yet (this task doesn't itself evaluate anything - see
    # queue_evaluation's docstring) would always report back an empty
    # updatedMoves, even though allowedMoves was just persisted for real.
    # dict.fromkeys: union, de-duplicated, order-preserving.
    result = entry.get("result")
    engine_moves = [m["move"] for m in result["topMoves"]] if result else []
    known_moves = list(dict.fromkeys([*engine_moves, *allowed_moves]))

    return {
        "solvedStatus": entry["bookSolvedStatus"],
        "updatedMoves": [{"move": m, "isAllowed": m in allowed_moves} for m in known_moves],
    }
