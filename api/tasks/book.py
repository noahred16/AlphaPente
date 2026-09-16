import threading
import uuid

from celery.signals import worker_ready
from rocksdict import Rdict

from api.celery_app import celery_app
from api.engine import run_search
from api.kv_store import (
    add_parent_edge,
    delete_position_if_orphaned,
    get_book_db,
    get_entry,
    propagate_book_status,
    register_queued_job,
    reset_queued_jobs,
    save_entry,
    unregister_queued_job,
)
from api.schemas.book import JobStatus, SearchLevel
from api.zobrist import compute_hash

# Hardcoded per-level iteration counts - the only search-depth control
# exposed to the UI (see EvaluateRequest.level). VERY_FAST/FAST/MEDIUM target
# real wall-clock times; DEEP targets the arena limit instead (900_000_000
# was this project's original always-on iteration count, run by hand and
# left to search until the tree arena filled up - not a real time budget).
# Calibrated against docker-compose.yml's worker config (NUM_THREADS=6,
# ARENA_SIZE_GB=29), measured directly on an ordinary early-game position
# with that config - per-iteration throughput isn't constant (thread
# startup/tree setup overhead dominates at low iteration counts), so each
# tier was timed on its own rather than scaled from another's sims/sec:
#   ./pente "K10 L9" 200000 -n -j     ->   200,000 sims in ~5s
#   ./pente "K10 L9" 2000000 -n -j    ->   2,000,000 sims in ~30s
#   ./pente "K10 L9" 3000000 -n -j    ->   3,000,000 sims in ~40s (used to extrapolate MEDIUM)
SEARCH_LEVEL_ITERATIONS: dict[SearchLevel, int] = {
    SearchLevel.VERY_FAST: 200_000,  # ~5s
    SearchLevel.FAST: 2_000_000,  # ~30s
    SearchLevel.MEDIUM: 9_000_000,  # ~2min
    SearchLevel.DEEP: 900_000_000,  # arena-bound, not time-bound
}

# The same empirical benchmarks above, as a rough wall-clock estimate rather
# than an iteration count - used only for the queue's ETA display (see
# estimated_seconds_for/QueuedJob.estimatedSeconds). DEEP has none: it's
# arena-bound, not time-bound, so there's nothing meaningful to estimate.
SEARCH_LEVEL_SECONDS: dict[SearchLevel, int | None] = {
    SearchLevel.VERY_FAST: 5,
    SearchLevel.FAST: 30,
    SearchLevel.MEDIUM: 120,
    SearchLevel.DEEP: None,
}


def estimated_seconds_for(target_visits: int) -> int | None:
    """Rough wall-clock estimate for a job dispatched at `target_visits` -
    see SEARCH_LEVEL_SECONDS. None if it matches DEEP, or (shouldn't happen
    for a job this app itself dispatched) no known level at all."""
    for level, iterations in SEARCH_LEVEL_ITERATIONS.items():
        if iterations == target_visits:
            return SEARCH_LEVEL_SECONDS[level]
    return None

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


@worker_ready.connect
def _reset_queue_registry_on_startup(**kwargs) -> None:
    """Clears the queue registry once, when the worker actually starts
    (rather than merely being imported, e.g. by tests) - see
    kv_store.reset_queued_jobs. Anything recorded as pending at that moment
    is necessarily stale: this process's own dispatch-then-acquire-then-
    finally sequence below is the only thing that ever writes an entry, and
    none of that survives a restart."""
    reset_queued_jobs(db=get_book_db())


@celery_app.task(name="book.evaluate")
def evaluate_position(moves: list[str], target_visits: int | None = None, db: Rdict | None = None) -> dict:
    """Run MCTS evaluation on a position and persist the result in book_db.
    Only one evaluate_position actually searches at a time - see
    _search_semaphore above - regardless of how many are dispatched
    concurrently; jobStatus is set to QUEUED immediately (before even
    acquiring the semaphore) and only flips to IN_PROGRESS once this one
    actually starts, not while merely waiting for a turn.

    That immediate QUEUED write matters beyond just being accurate: it's
    also what stops GET /pente/book (see get_book_entry's needs_evaluation)
    from re-dispatching the exact same still-waiting position on every poll.
    Real bug this reproduces - before this write existed, a position's
    persisted jobStatus stayed whatever it was *before* being queued (e.g.
    "IDLE" for a bare stub) for as long as it sat waiting behind the
    semaphore under any real backlog; every GET polled during that whole
    window saw that same stale status, concluded (wrongly) that nothing had
    been queued yet, and dispatched another evaluate_position for it - the
    same position piling up many duplicate entries in the queue (see
    GET /pente/book/queue), each duplicate making the real backlog longer
    and so the window for yet another duplicate even wider.

    `db` is a test-only escape hatch (defaults to the real book_db) - it
    isn't JSON-serializable, so never pass it through .delay()/apply_async();
    only call this directly (as tests do) with an explicit db.
    """
    db = db if db is not None else get_book_db()
    iterations = target_visits or SEARCH_LEVEL_ITERATIONS[SearchLevel.DEEP]
    save_entry(moves, job_status=JobStatus.QUEUED.value, db=db)

    # Recorded from the moment this is dispatched - before even acquiring
    # _search_semaphore below - so GET /pente/book/queue can show a job still
    # waiting its turn, not just one actually running. `job_id` only needs to
    # be unique per call, so a fresh uuid (not Celery's own task id) works
    # whether this came through Celery or - as in tests - a direct call.
    job_id = uuid.uuid4().hex
    register_queued_job(job_id, moves, iterations, db=db)
    try:
        _search_semaphore.acquire()
        save_entry(moves, job_status=JobStatus.IN_PROGRESS.value, target_visits=iterations, db=db)

        try:
            result = run_search(moves, iterations)
        except Exception:
            save_entry(moves, job_status=JobStatus.FAILED.value, db=db)
            raise
        finally:
            _search_semaphore.release()

        # A move the engine itself found and ranked is, by definition, in
        # scope for this position's proof - union it into allowedMoves rather
        # than leaving it to require a separate manual approval. Preserves
        # anything already there (e.g. added by hand before this search
        # finished, or kept from a previous search's allowedMoves) - this
        # only ever adds, it never drops a move someone already removed via a
        # later, smaller PUT.
        existing_allowed = (get_entry(compute_hash(moves), db=db) or {}).get("allowedMoves") or []
        engine_moves = [m["move"] for m in result["topMoves"]]
        merged_allowed = list(dict.fromkeys([*existing_allowed, *engine_moves]))

        hash_hex = save_entry(
            moves, job_status=JobStatus.IDLE.value, result=result, allowed_moves=merged_allowed, db=db
        )

        # Record this position as the parent of every candidate reply it
        # found, then recompute (and, if changed, propagate up) its own
        # book-level solved status - see kv_store.propagate_book_status.
        for top_move in result["topMoves"]:
            add_parent_edge(moves + [top_move["move"]], parent_hash=hash_hex, db=db)
        propagate_book_status(hash_hex, db=db)

        return result
    finally:
        unregister_queued_job(job_id, db=db)


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
