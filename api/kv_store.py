import json
import threading
from datetime import datetime, timezone

from rocksdict import AccessType, Rdict

from api.config import settings
from api.zobrist import apply_symmetry, compute_canonical_hash, compute_hash

_book_db: Rdict | None = None
_book_db_reader: Rdict | None = None

# Guards the actual book_db writes below (save_entry/add_parent_edge/
# add_canonical_member/propagate_book_status) against lost updates when two
# tasks run concurrently in the same worker process (e.g. --pool=threads) -
# a plain read-modify-write (read `existing`, merge, write back) can drop
# one side's change if two threads interleave it. Deliberately NOT guarding
# get_book_db()/get_book_db_reader()'s lazy open (a separate, accepted race)
# - this is scoped to writes only. RLock, not Lock: these functions call
# each other (and propagate_book_status recurses into itself) from the same
# thread, which a plain Lock would deadlock on.
_book_db_write_lock = threading.RLock()


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
    target_visits: int | None = None,
    db: Rdict | None = None,
) -> str:
    """Create or merge-update the book entry for the position reached by
    `moves`, keyed by its Zobrist hash. Only the book_db owner (the Celery
    worker) should call this without an explicit db - see get_book_db().

    job_status/result/allowed_moves/target_visits left as None preserve
    whatever's already stored for that field (or a sensible default if
    there's no existing entry) rather than overwriting it - e.g. marking a
    job IN_PROGRESS doesn't erase a previous result, and vice versa.
    date_started is set once (first write for a given hash) and always
    preserved after that. Returns the hash.
    """
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        hash_hex = compute_hash(moves)
        canonical_hash, sym = compute_canonical_hash(moves)

        existing = get_entry(hash_hex, db=db) or {}

        db[hash_hex] = json.dumps(
            {
                "moves": moves,
                "jobStatus": job_status if job_status is not None else existing.get("jobStatus", "IDLE"),
                "date_started": existing.get("date_started") or datetime.now(timezone.utc).isoformat(),
                "result": result if result is not None else existing.get("result"),
                "allowedMoves": allowed_moves if allowed_moves is not None else existing.get("allowedMoves"),
                # The iteration count this position's evaluation was actually
                # requested at (see api/tasks/book.py's SEARCH_LEVEL_ITERATIONS) -
                # not necessarily how many it actually ran, since a search can
                # finish early once it proves a result. Lets the frontend show
                # which level (Fast/Medium/Deep) a position was last searched
                # at, and offer re-running it deeper.
                "targetVisits": target_visits if target_visits is not None else existing.get("targetVisits"),
                # Graph bookkeeping for solved-status propagation - see
                # propagate_book_status(). Not settable via save_entry itself:
                # parentHashes only ever grows through add_parent_edge(), and
                # bookSolvedStatus is only ever recomputed through
                # compute_book_solved_status(), never hand-set here, so a plain
                # save_entry() call (e.g. just bumping jobStatus) can't
                # accidentally clobber either with a stale value.
                "parentHashes": existing.get("parentHashes", []),
                "bookSolvedStatus": existing.get("bookSolvedStatus", "UNSOLVED"),
                # Which symmetry group this exact physical orientation belongs to
                # - see add_canonical_member/get_group_status. Recomputed every
                # save (cheap, in-process) so it's never stale.
                "canonicalHash": canonical_hash,
                "sym": sym,
            }
        )
        add_canonical_member(canonical_hash, hash_hex, db=db)
        return hash_hex


def add_parent_edge(child_moves: list[str], parent_hash: str, db: Rdict | None = None) -> str:
    """Record that `parent_hash` has the position reached by `child_moves` as
    a child - creating a bare entry for it first if it doesn't exist yet, so
    the edge isn't lost even if the child itself has never been evaluated.
    Idempotent (parent_hash is only added once). Returns the child's hash."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        child_hash = save_entry(child_moves, db=db)  # no-op merge-update if it already exists

        entry = get_entry(child_hash, db=db)
        parents = entry.get("parentHashes", [])
        if parent_hash not in parents:
            entry["parentHashes"] = [*parents, parent_hash]
            db[child_hash] = json.dumps(entry)
        return child_hash


def add_canonical_member(canonical_hash: str, physical_hash: str, db: Rdict | None = None) -> None:
    """Record that `physical_hash` is one physical orientation belonging to
    the symmetry group `canonical_hash` (positions that are
    rotations/reflections of each other - see api/zobrist.py's
    compute_canonical_hash). Idempotent. Stored under a "canon:" prefix, a
    disjoint key namespace from physical hashes so the two never collide."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        key = f"canon:{canonical_hash}"
        raw = db.get(key)
        members = json.loads(raw) if raw is not None else []
        if physical_hash not in members:
            members.append(physical_hash)
            db[key] = json.dumps(members)


def get_canonical_members(canonical_hash: str, db: Rdict | None = None) -> list[str]:
    """Every physical hash known to belong to symmetry group `canonical_hash`."""
    db = db if db is not None else get_book_db_reader()
    if db is None:
        return []
    raw = db.get(f"canon:{canonical_hash}")
    return json.loads(raw) if raw is not None else []


def remove_canonical_member(canonical_hash: str, physical_hash: str, db: Rdict | None = None) -> None:
    """The inverse of add_canonical_member - drops `physical_hash` from its
    symmetry group's member list, e.g. once that entry is actually deleted
    (see delete_position_if_orphaned). A stale reference left behind here
    wouldn't corrupt anything on its own (get_group_status/find_resolved_twin
    already skip a member hash whose entry is missing), but there's no
    reason to let a dead hash accumulate in the index forever."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        key = f"canon:{canonical_hash}"
        raw = db.get(key)
        members = json.loads(raw) if raw is not None else []
        if physical_hash in members:
            members.remove(physical_hash)
            if members:
                db[key] = json.dumps(members)
            else:
                del db[key]


def delete_position_if_orphaned(hash_hex: str, parent_hash: str, db: Rdict | None = None) -> None:
    """Drop `parent_hash` from hash_hex's parentHashes; if that leaves it
    with no parents at all, the position is genuinely unreachable now (not a
    transposition some other position still legitimately reaches it through
    - see add_parent_edge, the only thing that ever adds a parentHashes
    entry) - so delete it outright, recursing into its own candidate
    children first (same check, so a grandchild only dies if it too has no
    other parent left) and dropping it from its canonical group's member
    index. A position with other parents remaining is just detached from
    this one, never deleted - its data may still be exactly what those other
    parents are relying on.

    No-op if hash_hex doesn't exist (e.g. a manually-added move that was
    never itself touched - there's nothing to delete)."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        entry = get_entry(hash_hex, db=db)
        if entry is None:
            return

        remaining_parents = [p for p in entry.get("parentHashes", []) if p != parent_hash]
        if remaining_parents:
            entry["parentHashes"] = remaining_parents
            db[hash_hex] = json.dumps(entry)
            return

        for move in entry.get("allowedMoves") or []:
            child_hash = compute_hash(entry["moves"] + [move])
            delete_position_if_orphaned(child_hash, parent_hash=hash_hex, db=db)

        remove_canonical_member(entry["canonicalHash"], hash_hex, db=db)
        del db[hash_hex]


def get_group_status(canonical_hash: str, db: Rdict, exclude_hash: str | None = None) -> str:
    """The best-known solved status shared by every physical orientation of
    this position (except `exclude_hash`, if given - see
    compute_book_solved_status, which must exclude the very entry it's
    recomputing a fresh value for, or it would just read back its own stale
    pre-recomputation status as if it were independent confirming evidence).
    A WIN/LOSS/DRAW proven via any one orientation's search is a fact about
    the actual position, not an artifact of which orientation was searched
    or which allowedMoves subset it used - so it's sound to treat it as
    proven for every symmetric twin too, regardless of whether those twins
    have been evaluated (or restricted) at all. UNSOLVED if no other member
    is resolved yet, or the group has no other members."""
    for member_hash in get_canonical_members(canonical_hash, db=db):
        if member_hash == exclude_hash:
            continue
        member = get_entry(member_hash, db=db)
        if member and member["bookSolvedStatus"] != "UNSOLVED":
            return member["bookSolvedStatus"]
    return "UNSOLVED"


def find_resolved_twin(canonical_hash: str, exclude_hash: str, db: Rdict | None = None) -> dict | None:
    """The first canonical-group member (other than `exclude_hash`) that has
    a completed engine result of its own, or None if no symmetric twin has
    been evaluated yet. Lets a query for a position that's never itself been
    searched answer instead by translating a twin's result - see
    translate_result()."""
    db = db if db is not None else get_book_db_reader()
    if db is None:
        return None
    for member_hash in get_canonical_members(canonical_hash, db=db):
        if member_hash == exclude_hash:
            continue
        member = get_entry(member_hash, db=db)
        if member and member.get("result") is not None:
            return member
    return None


def translate_result(result: dict, from_sym: int, to_sym: int) -> dict:
    """Re-express a result's move labels (bestMove and every topMoves[i].move)
    from one orientation's coordinate frame into another's. Each move passes
    through its shared canonical form: forward by `from_sym` to get there,
    then the inverse of `to_sym` to get back out into the target orientation
    - see api/zobrist.py's apply_symmetry."""

    def translate_move(move: str) -> str:
        canonical_move = apply_symmetry(move, from_sym)
        return apply_symmetry(canonical_move, to_sym, inverse=True)

    translated = dict(result)
    if translated.get("bestMove") is not None:
        translated["bestMove"] = translate_move(translated["bestMove"])
    translated["topMoves"] = [{**m, "move": translate_move(m["move"])} for m in result["topMoves"]]
    return translated


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
    if any(s == "SOLVED_WIN" for s in child_statuses):
        return "SOLVED_LOSS"
    if any(s == "UNSOLVED" for s in child_statuses):
        return "UNSOLVED"
    if all(s == "SOLVED_LOSS" for s in child_statuses):
        return "SOLVED_WIN"
    return "SOLVED_DRAW"  # no WIN, nothing UNSOLVED, not all LOSS -> a settled mix of LOSS/DRAW


def compute_book_solved_status(entry: dict, db: Rdict) -> str:
    """A node's book-level solved status, in priority order:

    1. Its own engine result, if that search already proved something outright.
    2. A symmetric twin's status (get_group_status) - a rotation/reflection
       of this exact position might already be resolved even though this
       orientation has never itself been evaluated.
    3. Otherwise, whatever can be concluded by combining candidate children's
       own book-level statuses (each one also checked via its own symmetry
       group, not just its exact physical hash - so a child that's only
       *itself* known through a symmetric twin still counts). This is what
       makes book-level status more powerful than result["solvedStatus"]
       alone - each child was evaluated in its own separate, bounded search
       with no way to know what any sibling's search found, so a forced
       sequence spanning several of them can go provable here even though no
       single search was deep enough to see it.

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

    own_hash = compute_hash(entry["moves"])
    twin_status = get_group_status(entry["canonicalHash"], db=db, exclude_hash=own_hash)
    if twin_status != "UNSOLVED":
        return twin_status

    allowed = entry.get("allowedMoves")
    if allowed is not None:
        candidate_moves = allowed
    elif result:
        candidate_moves = [m["move"] for m in result["topMoves"]]
    else:
        candidate_moves = []

    child_statuses = []
    for move in candidate_moves:
        child_canonical_hash, _ = compute_canonical_hash(entry["moves"] + [move])
        child_statuses.append(get_group_status(child_canonical_hash, db=db))

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
    this can't loop forever.

    Also walks every symmetric twin's parents (not the twins themselves -
    get_group_status already surfaces this node's new status to them without
    needing their own bookSolvedStatus field rewritten). Necessary because a
    twin can be reached from a completely different parent than this node
    was - without this, that parent would only find out about a proof
    borrowed through its child's twin the next time something else happened
    to trigger a recompute of it directly.
    """
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
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

        for twin_hash in get_canonical_members(entry["canonicalHash"], db=db):
            if twin_hash == hash_hex:
                continue
            twin_entry = get_entry(twin_hash, db=db)
            if twin_entry:
                for parent_hash in twin_entry.get("parentHashes", []):
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


# Single key holding the whole evaluate_position queue registry (see
# register_queued_job) - "queue:" is a disjoint namespace from both physical
# position hashes (fixed-length hex Zobrist hashes - see compute_hash) and
# "canon:"-prefixed symmetry-group keys (see add_canonical_member), so none
# of the three can ever collide.
_QUEUE_REGISTRY_KEY = "queue:jobs"


def register_queued_job(job_id: str, moves: list[str], target_visits: int, db: Rdict | None = None) -> None:
    """Record an evaluate_position call as pending, from the moment it's
    dispatched - before it even acquires api/tasks/book.py's
    _search_semaphore, so a job still waiting its turn shows up here just
    like one actually running - until unregister_queued_job removes it.
    `job_id` just needs to be unique per call (evaluate_position generates
    its own, independent of Celery's own task id, so this works whether the
    call came through Celery or - as in tests - directly).

    Lets the API report a live list of what's queued/running and how deep
    each one is set to search (see GET /pente/book/queue) without touching
    Celery's own broker at all - the queue's real state already lives here,
    in the one place (book_db) the API otherwise reads everything from.
    """
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        registry = _load_queue_registry(db)
        registry[job_id] = {
            "moves": moves,
            "targetVisits": target_visits,
            "queuedAt": datetime.now(timezone.utc).isoformat(),
        }
        db[_QUEUE_REGISTRY_KEY] = json.dumps(registry)


def unregister_queued_job(job_id: str, db: Rdict | None = None) -> None:
    """Remove a job recorded by register_queued_job, once it's finished -
    successfully or not (see evaluate_position's outer finally)."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        registry = _load_queue_registry(db)
        if registry.pop(job_id, None) is not None:
            db[_QUEUE_REGISTRY_KEY] = json.dumps(registry)


def get_queued_jobs(db: Rdict | None = None) -> list[dict]:
    """Every job currently recorded as pending (see register_queued_job),
    oldest first. Defaults to the read-only secondary handle, like get_entry."""
    db = db if db is not None else get_book_db_reader()
    if db is None:
        return []
    return sorted(_load_queue_registry(db).values(), key=lambda job: job["queuedAt"])


def reset_queued_jobs(db: Rdict | None = None) -> None:
    """Clears the queue registry outright - called once when the worker
    process starts (see the worker_ready signal handler in
    api/tasks/book.py), since nothing recorded as in-flight at that moment
    actually survived whatever just (re)started the process."""
    db = db if db is not None else get_book_db()
    with _book_db_write_lock:
        db[_QUEUE_REGISTRY_KEY] = json.dumps({})


def _load_queue_registry(db: Rdict) -> dict:
    raw = db.get(_QUEUE_REGISTRY_KEY)
    return json.loads(raw) if raw is not None else {}
