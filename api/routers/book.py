from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.kv_store import find_resolved_twin, get_entry, get_queued_jobs, translate_result
from api.schemas.book import (
    AllowedMovesRequest,
    AllowedMovesResponse,
    BookEntry,
    EvaluateRequest,
    EvaluateResponse,
    JobStatus,
    MoveStr,
    QueuedJob,
    QueueResponse,
    SearchLevel,
    SolvedStatus,
    TopMove,
)
from api.tasks.book import SEARCH_LEVEL_ITERATIONS, estimated_seconds_for, evaluate_position, set_allowed_moves
from api.zobrist import compute_canonical_hash, compute_hash

router = APIRouter(prefix="/pente/book", tags=["book"])


def _to_book_entry(entry: dict) -> BookEntry:
    """Map a stored book_db entry ({moves, jobStatus, date_started, result})
    onto the API's BookEntry shape. `result` is the engine's own JSON output
    (see GameUtils::runSearchAndReportJSON) - None if no search has completed
    yet (freshly queued/in-progress, or never evaluated at all). topMoves
    doesn't wait on that, though: allowedMoves (see PUT /allowed-moves) seeds
    the move list regardless, so moves you already know you want to track
    show up - and can be built on immediately via POST - without waiting for
    a search to finish."""
    result = entry.get("result")
    # None (never set via PUT /allowed-moves) means "nothing restricted yet" - allow everything.
    allowed_moves = entry.get("allowedMoves")

    def child_entry_for(move: str) -> dict | None:
        return get_entry(compute_hash(entry["moves"] + [move]))

    def expanded_state(move: str) -> str:
        """Whether `move`'s own child position has already been searched -
        found via its own book_db entry, not anything about this (the
        parent's) search. Matches TopMove.expanded's ExpandedState values.

        Deliberately reports "true" (not "in progress") for a position with
        both a real result *and* a currently-running job - a re-run/deepen
        of an already-searched move - since a still-good older result is
        more useful to show than blanking it out while the new one cooks.
        See child_in_progress for the raw, unmasked signal instead."""
        child = child_entry_for(move)
        if child is None:
            return "false"
        if child.get("result") is not None:
            return "true"
        if child.get("jobStatus") in ("QUEUED", "IN_PROGRESS"):
            return "in progress"
        return "false"  # a bare add_parent_edge stub - recorded as a child, but never queued

    def child_in_progress(move: str) -> bool:
        """Whether `move`'s own child position has a real search job running
        right now - see TopMove.childInProgress."""
        child = child_entry_for(move)
        return child is not None and child.get("jobStatus") in ("QUEUED", "IN_PROGRESS")

    def child_move_count(move: str) -> int:
        """How many moves are already in the allowed-move list of the
        position `move` leads to - 0 if that child has never been touched."""
        child = child_entry_for(move)
        return len(child.get("allowedMoves") or []) if child else 0

    def child_target_visits(move: str) -> int | None:
        """The targetVisits `move`'s own last evaluation was run at - None if
        it's never been evaluated. See TopMove.childTargetVisits."""
        child = child_entry_for(move)
        return child.get("targetVisits") if child else None

    def child_best_move_value(move: str) -> float | None:
        """The avgValue of `move`'s own child's best reply, per its own
        independent search - see TopMove.childBestMoveValue. None if that
        search hasn't produced a result yet, or found no legal replies."""
        child = child_entry_for(move)
        result = child.get("result") if child else None
        best_move = result.get("bestMove") if result else None
        if not best_move:
            return None
        return next((tm["avgValue"] for tm in result["topMoves"] if tm["move"] == best_move), None)

    def child_status(move: str, engine_status: str) -> str:
        """The freshest known solved status of the position `move` leads to.
        `bookSolvedStatus` uses the exact same "last mover" convention as a
        topMove's own `status` (see ParallelMCTS::backpropagate) - it's just
        potentially fresher, since it can reflect an independent deep search
        of that exact child (or one propagated from a symmetric twin) that
        happened after - or entirely outside - this position's own bounded
        search. `engine_status` (this search's own one-ply read on the move,
        or "UNSOLVED" for a move it never itself explored) is the fallback
        when the child hasn't been touched, or hasn't resolved anything
        beyond what this search already found on its own."""
        child = child_entry_for(move)
        child_book_status = child.get("bookSolvedStatus", "UNSOLVED") if child else "UNSOLVED"
        return child_book_status if child_book_status != "UNSOLVED" else engine_status

    # Two "different" candidate moves from this position can be board-
    # symmetric twins of each other (their child positions share a canonical
    # hash - see kv_store.compute_canonical_hash) even though nothing else
    # here treats them as duplicates. Flag every move after the first seen
    # in its symmetry class, rather than merging/hiding either one - keeps
    # all engine data visible while telling the caller (and the
    # allowed-moves proof logic) that resolving one resolves both.
    seen_canonical: dict[str, str] = {}

    def symmetric_to(move: str) -> str | None:
        canonical_hash, _ = compute_canonical_hash(entry["moves"] + [move])
        if canonical_hash in seen_canonical:
            return seen_canonical[canonical_hash]
        seen_canonical[canonical_hash] = move
        return None

    top_moves = [
        TopMove(
            move=m["move"],
            visits=m["visits"],
            prior=m["prior"],
            avgValue=m["avgValue"],
            puct=m["puct"],
            status=child_status(m["move"], m["status"]),
            isAllowed=allowed_moves is None or m["move"] in allowed_moves,
            expanded=expanded_state(m["move"]),
            symmetricTo=symmetric_to(m["move"]),
            childMoveCount=child_move_count(m["move"]),
            childTargetVisits=child_target_visits(m["move"]),
            childInProgress=child_in_progress(m["move"]),
            childBestMoveValue=child_best_move_value(m["move"]),
        )
        for m in (result["topMoves"] if result else [])
    ]

    # A move can be force-included in allowedMoves even though the engine's
    # own search never ranked it highly enough to appear in topMoves (see
    # kv_store.compute_book_solved_status's docstring on why that's allowed) -
    # or hasn't run at all yet - surface those too, with no engine stats to
    # show (avgValue=None - the frontend renders that as "N/A"), so they're
    # visible and queueable instead of only ever affecting solved-status
    # bookkeeping.
    known_moves = {m.move for m in top_moves}
    top_moves += [
        TopMove(
            move=move,
            visits=0,
            prior=0.0,
            avgValue=None,
            puct=None,
            status=child_status(move, SolvedStatus.UNSOLVED.value),
            isAllowed=True,
            expanded=expanded_state(move),
            symmetricTo=symmetric_to(move),
            childMoveCount=child_move_count(move),
            childTargetVisits=child_target_visits(move),
            childInProgress=child_in_progress(move),
            childBestMoveValue=child_best_move_value(move),
        )
        for move in (allowed_moves or [])
        if move not in known_moves
    ]

    return BookEntry(
        moves=entry["moves"],
        jobStatus=entry["jobStatus"],
        totalVisits=result["totalVisits"] if result else 0,
        # bookSolvedStatus (propagated across separately-evaluated children,
        # restricted to allowedMoves - see kv_store.propagate_book_status),
        # not result["solvedStatus"] (that one only reflects what this one
        # bounded search found on its own).
        solvedStatus=entry.get("bookSolvedStatus", "UNSOLVED"),
        bestValue=result["rootAvgValue"] if result else 0.0,
        bestMove=result["bestMove"] if result else None,
        date_started=entry["date_started"],
        topMoves=top_moves,
        targetVisits=entry.get("targetVisits"),
    )


@router.get("", response_model=BookEntry)
def get_book_entry(moves: list[MoveStr] = Query(default=[]), level: SearchLevel = SearchLevel.MEDIUM) -> BookEntry:
    """Return move evaluation and opening book state for a position.

    If this exact orientation has no result of its own, but a board-
    symmetric twin (a rotation/reflection - see kv_store.compute_canonical_hash)
    has already been searched, answers with that twin's result translated
    into this orientation's coordinate frame instead of re-running a
    redundant search. Only when no twin exists either does it queue a fresh
    evaluation (at `level`) and return a freshly-QUEUED entry rather than
    404ing. `level` only matters the first time a position is seen - it has
    no effect once an entry (even a bare, unevaluated one) already exists;
    re-running an existing position at a deeper level is POST's job.
    """
    try:
        hash_hex = compute_hash(moves)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    entry = get_entry(hash_hex)
    if entry is None or entry.get("result") is None:
        canonical_hash, sym = compute_canonical_hash(moves)
        twin = find_resolved_twin(canonical_hash, exclude_hash=hash_hex)
        if twin is not None:
            base = entry or {}
            return _to_book_entry(
                {
                    "moves": moves,
                    "jobStatus": base.get("jobStatus", twin["jobStatus"]),
                    "date_started": base.get("date_started", twin["date_started"]),
                    "result": translate_result(twin["result"], twin["sym"], sym),
                    "allowedMoves": base.get("allowedMoves"),
                    "bookSolvedStatus": twin["bookSolvedStatus"],
                }
            )

    # Not just "entry is None": PUT /allowed-moves (and add_parent_edge)
    # create a bare entry - jobStatus "IDLE", no result - for a position
    # that's never actually been queued. Without also catching that case
    # here, such a position could never get evaluated at all: this is the
    # only place that queues one, and "entry is None" alone would never
    # fire again once that bare entry exists. FAILED is included too, so a
    # crashed run gets retried on the next GET rather than staying stuck.
    needs_evaluation = entry is None or (
        entry.get("result") is None and entry.get("jobStatus") not in ("QUEUED", "IN_PROGRESS")
    )
    if needs_evaluation:
        evaluate_position.delay(moves, target_visits=SEARCH_LEVEL_ITERATIONS[level])
        entry = {
            **(entry or {}),
            "moves": moves,
            "jobStatus": JobStatus.QUEUED.value,
            "date_started": (entry or {}).get("date_started") or datetime.now(timezone.utc).isoformat(),
            "result": None,
        }

    return _to_book_entry(entry)


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

    task = evaluate_position.delay(body.moves, target_visits=SEARCH_LEVEL_ITERATIONS[body.level])
    return EvaluateResponse(job_id=task.id, jobStatus=JobStatus.QUEUED)


@router.get("/queue", response_model=QueueResponse)
def get_queue() -> QueueResponse:
    """Every evaluate_position job dispatched but not yet finished - the one
    actually searching right now (there's no reliable way to tell which
    entry that is - see kv_store.register_queued_job) included, same as
    everything still waiting its turn - oldest first.

    Global, not scoped to any one position - that's why this isn't just
    another field on GET /pente/book (which is), and why the frontend only
    fetches it for its own separate queue panel rather than on every poll.
    """
    jobs = [
        QueuedJob(**job, estimatedSeconds=estimated_seconds_for(job["targetVisits"])) for job in get_queued_jobs()
    ]
    return QueueResponse(count=len(jobs), jobs=jobs)


@router.put("/allowed-moves", response_model=AllowedMovesResponse)
def update_allowed_moves(body: AllowedMovesRequest) -> AllowedMovesResponse:
    """Persist the allowed-move subset for a position.

    Like POST, this is a write and can't touch book_db directly from this
    process (see queue_evaluation's docstring) - it runs via a Celery task.
    Unlike POST, it waits synchronously for that task rather than returning
    a job_id: the response needs the task's result (solvedStatus,
    updatedMoves), and persisting an allowed-move subset plus propagating
    any solved-status change is fast, unlike an actual MCTS search.
    """
    try:
        compute_hash(body.moves)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    result = set_allowed_moves.delay(body.moves, body.allowedMoves).get(timeout=10)
    return AllowedMovesResponse(**result)
