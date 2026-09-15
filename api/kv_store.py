import json
from datetime import datetime, timezone

from rocksdict import AccessType, Rdict

from api.config import settings
from api.zobrist import compute_hash

_book_db: Rdict | None = None
_book_db_reader: Rdict | None = None


def get_book_db() -> Rdict:
    """Lazily open the real book_db for read-write, not at import time.

    Only the process that owns writes (the Celery worker) should call this:
    RocksDB allows only one read-write handle on a given path at a time -
    a second process opening it this way raises an IO error ("lock hold by
    current process"). See get_book_db_reader() for the concurrent-safe
    counterpart the API process uses instead.
    """
    global _book_db
    if _book_db is None:
        _book_db = Rdict(settings.book_db_path)
    return _book_db


def get_book_db_reader() -> Rdict | None:
    """Lazily open book_db as a RocksDB "secondary" instance: unlike
    get_book_db(), this is safe to hold open in a different process from
    the one that owns the read-write handle (e.g. the API process reading
    while a separate Celery worker process writes). Calls
    try_catch_up_with_primary() on every use so reads reflect the writer's
    latest state rather than a stale snapshot from when this was first
    opened. Returns None if book_db doesn't exist yet (nothing written).
    """
    global _book_db_reader
    if _book_db_reader is None:
        try:
            _book_db_reader = Rdict(
                settings.book_db_path,
                access_type=AccessType.secondary(f"{settings.book_db_path}-secondary"),
            )
        except Exception:
            return None
    _book_db_reader.try_catch_up_with_primary()
    return _book_db_reader


def save_entry(
    moves: list[str],
    job_status: str | None = None,
    result: dict | None = None,
    allowed_moves: list[str] | None = None,
    db: Rdict | None = None,
) -> str:
    """Create or merge-update the book entry for the position reached by
    `moves`, keyed by its Zobrist hash. Only the book_db owner (the Celery
    worker) should call this without an explicit db - see get_book_db().

    job_status/result/allowed_moves left as None preserve whatever's already
    stored for that field (or a sensible default if there's no existing
    entry) rather than overwriting it - e.g. marking a job IN_PROGRESS
    doesn't erase a previous result, and vice versa. date_started is set
    once (first write for a given hash) and always preserved after that.
    Returns the hash.
    """
    db = db if db is not None else get_book_db()
    hash_hex = compute_hash(moves)

    existing = get_entry(hash_hex, db=db) or {}

    db[hash_hex] = json.dumps(
        {
            "moves": moves,
            "jobStatus": job_status if job_status is not None else existing.get("jobStatus", "IDLE"),
            "date_started": existing.get("date_started") or datetime.now(timezone.utc).isoformat(),
            "result": result if result is not None else existing.get("result"),
            "allowedMoves": allowed_moves if allowed_moves is not None else existing.get("allowedMoves"),
            # Graph bookkeeping for solved-status propagation - see
            # propagate_book_status(). Not settable via save_entry itself:
            # parentHashes only ever grows through add_parent_edge(), and
            # bookSolvedStatus is only ever recomputed through
            # compute_book_solved_status(), never hand-set here, so a plain
            # save_entry() call (e.g. just bumping jobStatus) can't
            # accidentally clobber either with a stale value.
            "parentHashes": existing.get("parentHashes", []),
            "bookSolvedStatus": existing.get("bookSolvedStatus", "UNSOLVED"),
        }
    )
    return hash_hex


def add_parent_edge(child_moves: list[str], parent_hash: str, db: Rdict | None = None) -> str:
    """Record that `parent_hash` has the position reached by `child_moves` as
    a child - creating a bare entry for it first if it doesn't exist yet, so
    the edge isn't lost even if the child itself has never been evaluated.
    Idempotent (parent_hash is only added once). Returns the child's hash."""
    db = db if db is not None else get_book_db()
    child_hash = save_entry(child_moves, db=db)  # no-op merge-update if it already exists

    entry = get_entry(child_hash, db=db)
    parents = entry.get("parentHashes", [])
    if parent_hash not in parents:
        entry["parentHashes"] = [*parents, parent_hash]
        db[child_hash] = json.dumps(entry)
    return child_hash


def combine_child_statuses(child_statuses: list[str]) -> str:
    """Negamax over allowed children's solved statuses, in the same "last
    mover won" convention ParallelMCTS::backpropagate uses (see its comment:
    "SOLVED_WIN means the last mover won [...] a win from the parent's
    perspective") - a child WIN flips the parent to LOSS (the mover who
    reached it walked into a winning reply); a parent is only WIN once every
    child is LOSS (every reply is bad for whoever's replying); an unresolved
    child blocks any conclusion (returns UNSOLVED) even if a DRAW or LOSS is
    also present, since it might turn out to be the WIN that changes
    everything - only once every child is settled does a remaining DRAW
    resolve the parent to DRAW."""
    if not child_statuses:
        return "UNSOLVED"
    if any(s == "WIN" for s in child_statuses):
        return "LOSS"
    if any(s == "UNSOLVED" for s in child_statuses):
        return "UNSOLVED"
    if all(s == "LOSS" for s in child_statuses):
        return "WIN"
    return "DRAW"  # no WIN, nothing UNSOLVED, not all LOSS -> a settled mix of LOSS/DRAW


def compute_book_solved_status(entry: dict, db: Rdict) -> str:
    """A node's book-level solved status: its own engine result if that
    search already proved something outright, otherwise whatever can be
    concluded by combining its candidate children's own book-level statuses.
    The child fallback is what makes this different from (and more powerful
    than) result["solvedStatus"] alone - each child was evaluated in its own
    separate, bounded search that has no way to know what any of its
    siblings' searches found, so a forced sequence spanning several of them
    can go provable here even though no single search was deep enough to see
    it.

    Candidate children are allowedMoves directly when it's been set (an
    authoritative, human-curated list - NOT filtered against topMoves, since
    the whole point is to let someone force in a move the engine's own
    search didn't rank highly enough to include there) - falling back to
    every move in topMoves when it hasn't been.

    Always recomputed from the current allowed set and current children,
    never patched - see propagate_book_status().
    """
    result = entry.get("result")
    if result and result["solvedStatus"] != "UNSOLVED":
        return result["solvedStatus"]

    allowed = entry.get("allowedMoves")
    if allowed is not None:
        candidate_moves = allowed
    elif result:
        candidate_moves = [m["move"] for m in result["topMoves"]]
    else:
        candidate_moves = []

    child_statuses = []
    for move in candidate_moves:
        child_hash = compute_hash(entry["moves"] + [move])
        child_entry = get_entry(child_hash, db=db)
        child_statuses.append(child_entry["bookSolvedStatus"] if child_entry else "UNSOLVED")

    return combine_child_statuses(child_statuses)


def propagate_book_status(hash_hex: str, db: Rdict | None = None) -> None:
    """Recompute hash_hex's bookSolvedStatus and, only if it actually
    changed, recurse into every recorded parent (see add_parent_edge) to do
    the same - a change (in either direction: newly proved, or reverted back
    to UNSOLVED because a new unresolved child was added to allowedMoves)
    can flip an ancestor's own conclusion, however far up the DAG. Stops the
    instant a level doesn't change, or there are no more parents. No cycle
    risk: capture counts (baked into the hash) never decrease and every move
    adds exactly one stone, so the position graph is provably acyclic -
    this can't loop forever."""
    db = db if db is not None else get_book_db()
    entry = get_entry(hash_hex, db=db)
    if entry is None:
        return

    new_status = compute_book_solved_status(entry, db=db)
    if new_status == entry["bookSolvedStatus"]:
        return

    entry["bookSolvedStatus"] = new_status
    db[hash_hex] = json.dumps(entry)

    for parent_hash in entry.get("parentHashes", []):
        propagate_book_status(parent_hash, db=db)


def get_entry(hash_hex: str, db: Rdict | None = None) -> dict | None:
    """Look up the book entry stored for `hash_hex`, or None if nothing is
    stored for it. Defaults to the read-only secondary handle
    (get_book_db_reader), safe to call from the API process even while a
    separate worker process holds book_db open for writes.
    """
    db = db if db is not None else get_book_db_reader()
    if db is None:
        return None
    raw = db.get(hash_hex)
    return json.loads(raw) if raw is not None else None
