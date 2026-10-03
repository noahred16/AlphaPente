"""Tests for solved-status propagation up the parent DAG (see
kv_store.combine_child_statuses / propagate_book_status). Uses a temporary
RocksDB (pytest's tmp_path) so this never touches the real book_db, and
fabricated result dicts (no real engine search) since propagation only cares
about solvedStatus/topMoves, not how a position was actually reached.

Move labels below still have to be real, legal Pente coordinates, even
though the results themselves are fabricated - compute_hash always replays
them through the real engine (that's the whole point of it), and an
invalid-format label crashes it rather than raising cleanly (a real gap in
PenteGame::makeMove/parseMove, worth hardening separately - a
validly-formatted-but-illegal move like a repeated cell does raise
ValueError correctly, see test_zobrist.py)."""
import pytest
from rocksdict import Rdict

kv_store = pytest.importorskip("api.kv_store", reason="pente_native not built; see README for build steps")


@pytest.fixture
def db(tmp_path):
    rdict = Rdict(str(tmp_path / "test_book.db"))
    yield rdict
    rdict.close()


# ─── combine_child_statuses: pure negamax, no book_db involved ───────────────


@pytest.mark.parametrize(
    "child_statuses,expected",
    [
        ([], "UNSOLVED"),
        (["SOLVED_WIN"], "SOLVED_LOSS"),
        (["SOLVED_LOSS", "SOLVED_WIN", "UNSOLVED"], "SOLVED_LOSS"),  # one WIN is enough, regardless of the rest
        (["SOLVED_LOSS", "SOLVED_LOSS"], "SOLVED_WIN"),  # every reply bad for the opponent
        (["SOLVED_LOSS", "UNSOLVED"], "UNSOLVED"),  # unresolved blocks a conclusion even with no WIN
        (["SOLVED_LOSS", "SOLVED_DRAW"], "SOLVED_DRAW"),  # settled mix of LOSS/DRAW, no WIN, nothing pending
    ],
)
def test_combine_child_statuses(child_statuses, expected):
    assert kv_store.combine_child_statuses(child_statuses) == expected


# ─── add_parent_edge ──────────────────────────────────────────────────────────


def test_add_parent_edge_creates_bare_child_and_records_parent(db):
    child_hash = kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent1", db=db)

    entry = kv_store.get_entry(child_hash, db=db)
    assert entry["parentHashes"] == ["parent1"]
    assert entry["result"] is None  # bare - never evaluated


def test_add_parent_edge_is_idempotent_and_supports_multiple_parents(db):
    child_hash = kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent1", db=db)
    kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent1", db=db)  # duplicate, no-op
    kv_store.add_parent_edge(["K10", "L9"], parent_hash="parent2", db=db)  # a genuine transposition

    assert kv_store.get_entry(child_hash, db=db)["parentHashes"] == ["parent1", "parent2"]


# ─── propagate_book_status: the end-to-end scenario ──────────────────────────


def test_propagation_proves_up_two_levels_then_reverts_on_a_new_unresolved_child(db):
    """Grandparent G -> Parent P -> children A, B. Proves A and B as LOSS,
    which should prove P WIN and, in turn, G LOSS. Then widens P's
    allowedMoves to include a brand-new move "J10" that was never evaluated
    (not even in P's own topMoves - see compute_book_solved_status's
    docstring on why allowedMoves is authoritative, not just a topMoves
    filter) and confirms P - and G, cascading further up - correctly revert
    to UNSOLVED rather than keeping a now-unsound conclusion."""
    g_moves = []
    p_moves = ["K10"]
    a_moves = ["K10", "L9"]
    b_moves = ["K10", "L8"]

    g_hash = kv_store.save_entry(g_moves, result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "K10"}]}, db=db)
    p_hash = kv_store.save_entry(
        p_moves, result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "L9"}, {"move": "L8"}]}, db=db
    )
    kv_store.add_parent_edge(p_moves, parent_hash=g_hash, db=db)

    # Nothing's resolved yet.
    kv_store.propagate_book_status(g_hash, db=db)
    assert kv_store.get_entry(g_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"

    # Prove A as LOSS - not enough on its own (B still unresolved).
    a_hash = kv_store.add_parent_edge(a_moves, parent_hash=p_hash, db=db)
    kv_store.save_entry(a_moves, result={"solvedStatus": "SOLVED_LOSS", "topMoves": []}, db=db)
    kv_store.propagate_book_status(a_hash, db=db)
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"

    # Prove B as LOSS too - now every child of P is LOSS, so P proves WIN,
    # which in turn proves G LOSS (any WIN among G's children -> LOSS).
    b_hash = kv_store.add_parent_edge(b_moves, parent_hash=p_hash, db=db)
    kv_store.save_entry(b_moves, result={"solvedStatus": "SOLVED_LOSS", "topMoves": []}, db=db)
    kv_store.propagate_book_status(b_hash, db=db)
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "SOLVED_WIN"
    assert kv_store.get_entry(g_hash, db=db)["bookSolvedStatus"] == "SOLVED_LOSS"

    # Now add a brand-new, never-evaluated candidate "J10" to P's allowed set.
    kv_store.save_entry(p_moves, allowed_moves=["L9", "L8", "J10"], db=db)
    kv_store.propagate_book_status(p_hash, db=db)

    # P can no longer claim WIN - J10 is unresolved and might refute it - and
    # that reversion must cascade back up to G too, not just stop at P.
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"
    assert kv_store.get_entry(g_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"


def test_propagation_proves_a_win_from_moves_never_independently_expanded(db):
    """Real bug this reproduces: a single massive search can fully prove
    several of its own replies as SOLVED_LOSS internally (result["topMoves"])
    without any of them ever becoming their own independently-searched
    position - get_group_status alone has nothing to find for a move like
    that (no book_db entry of its own exists at all, or it's still just a
    bare, untouched stub from add_parent_edge), so compute_book_solved_status
    used to treat "never independently searched" as "unproven" instead of
    falling back to this search's own already-settled one-ply read, and the
    position stayed UNSOLVED forever despite every one of its own replies
    already being a proven loss."""
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",  # this search's own bounded run never itself proved the root
        "topMoves": [
            {"move": "L9", "status": "SOLVED_LOSS"},
            {"move": "L8", "status": "SOLVED_LOSS"},
            {"move": "M11", "status": "SOLVED_LOSS"},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    # None of L9/L8/M11 has ever been independently touched - no add_parent_edge,
    # no evaluate_position, no entry of any kind.

    kv_store.propagate_book_status(hash_hex, db=db)

    assert kv_store.get_entry(hash_hex, db=db)["bookSolvedStatus"] == "SOLVED_WIN"


# ─── compute_book_best_move / bookBestMove propagation ───────────────────────


def test_compute_book_best_move_picks_the_highest_value_unsolved_candidate(db):
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.2},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.7},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)

    best_move, best_value = kv_store.compute_book_best_move(kv_store.get_entry(hash_hex, db=db), db=db)

    assert best_move == "L8"
    # Flipped from L8's own avgValue (0.7): compute_book_best_move's returned
    # value matches result["rootAvgValue"]'s convention (whoever just moved
    # *into* this position), not the perspective used to rank candidates
    # against each other - see the function's own docstring.
    assert best_value == -0.7


def test_compute_book_best_move_prefers_a_proven_win_over_a_higher_value_unsolved_move(db):
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.9},
            {"move": "L8", "status": "SOLVED_WIN", "avgValue": 0.1},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)

    best_move, _ = kv_store.compute_book_best_move(kv_store.get_entry(hash_hex, db=db), db=db)

    assert best_move == "L8"


def test_compute_book_best_move_prefers_a_childs_own_deeper_value_over_the_stale_one_ply_read(db):
    """Real bug this reproduces: L8 looks better by the parent's own one-ply
    read alone (0.2 > 0.1), but L9's own child has since been independently,
    deeply searched and found to be much better once you look one ply
    further - a fact result["bestMove"] (the parent's frozen snapshot) has
    no way to reflect."""
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.1},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.2},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    kv_store.save_entry(
        moves + ["L9"],
        result={"solvedStatus": "UNSOLVED", "bestMove": "M7", "topMoves": [{"move": "M7", "avgValue": -0.9}]},
        db=db,
    )

    best_move, best_value = kv_store.compute_book_best_move(kv_store.get_entry(hash_hex, db=db), db=db)

    assert best_move == "L9"
    assert best_value == pytest.approx(-0.9)  # flipped - see the other test's comment above


def test_compute_book_best_move_prefers_a_childs_own_live_best_value_over_its_frozen_snapshot(db):
    """Real bug this reproduces, one recursion level deeper than the test
    above: L9's own child (a grandchild of K10) has itself been searched
    deeper since L9's own last search, and propagate_book_status already
    recomputed L9's own live bookBestMove/bookBestValue to reflect that - but
    value_for was still reading L9's own frozen result["bestMove"] snapshot
    instead of that live, more-informed number, so K10's own ranking never
    found out. Numbers chosen so the two methods disagree on the winner
    outright (not just the margin): L9 looks only mildly good by its own
    frozen one-ply read (0.05, worse than L8's 0.2) but is actually far
    better once its own child's deeper search is taken into account (0.9)."""
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.1},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.2},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    l9_hash = kv_store.save_entry(
        moves + ["L9"],
        result={"solvedStatus": "UNSOLVED", "bestMove": "M7", "topMoves": [{"move": "M7", "avgValue": 0.05}]},
        db=db,
    )
    kv_store.save_entry(
        moves + ["L9", "M7"],
        result={"solvedStatus": "UNSOLVED", "bestMove": "N6", "topMoves": [{"move": "N6", "avgValue": 0.9}]},
        db=db,
    )
    kv_store.propagate_book_status(l9_hash, db=db)  # L9's own bookBestMove/bookBestValue now reflect M7's deeper value

    best_move, best_value = kv_store.compute_book_best_move(kv_store.get_entry(hash_hex, db=db), db=db)

    assert best_move == "L9"
    assert best_value == pytest.approx(-0.9)


def test_propagate_book_status_updates_best_move_when_it_is_removed(db):
    """Issue this reproduces: removing the current best move used to leave
    the parent's bestMove pointing at a move that's no longer even allowed."""
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.9},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.1},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    kv_store.propagate_book_status(hash_hex, db=db)
    assert kv_store.get_entry(hash_hex, db=db)["bookBestMove"] == "L9"

    kv_store.save_entry(moves, allowed_moves=["L8"], db=db)  # L9 removed
    kv_store.propagate_book_status(hash_hex, db=db)

    assert kv_store.get_entry(hash_hex, db=db)["bookBestMove"] == "L8"


def test_propagate_book_status_updates_parent_best_move_when_a_childs_own_value_improves(db):
    """The other issue this fixes: deepening/re-running a child's own search
    can reveal it's actually the better choice, but propagate_book_status
    only used to recompute bookSolvedStatus - bookBestMove needs the exact
    same recompute-and-propagate-to-parents treatment, or a parent never
    finds out."""
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.1},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.2},
        ],
    }
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    l9_hash = kv_store.add_parent_edge(moves + ["L9"], parent_hash=hash_hex, db=db)
    kv_store.add_parent_edge(moves + ["L8"], parent_hash=hash_hex, db=db)
    kv_store.propagate_book_status(hash_hex, db=db)
    assert kv_store.get_entry(hash_hex, db=db)["bookBestMove"] == "L8"

    # L9's own child gets independently, deeply re-evaluated - propagating
    # from *there* (not from the parent) must still update the parent.
    kv_store.save_entry(
        moves + ["L9"],
        result={"solvedStatus": "UNSOLVED", "bestMove": "M7", "topMoves": [{"move": "M7", "avgValue": -0.9}]},
        db=db,
    )
    kv_store.propagate_book_status(l9_hash, db=db)

    assert kv_store.get_entry(hash_hex, db=db)["bookBestMove"] == "L9"


# ─── find_deepest_promising_leaf: the "Depth Search" button's own traversal ──


def test_find_deepest_promising_leaf_returns_moves_unchanged_for_a_never_seen_position(db):
    leaf = kv_store.find_deepest_promising_leaf(["K10"], db=db)

    assert leaf == ["K10"]


def test_find_deepest_promising_leaf_descends_one_level_to_an_unexpanded_child(db):
    moves = ["K10"]
    result = {
        "solvedStatus": "UNSOLVED",
        "topMoves": [
            {"move": "L9", "status": "UNSOLVED", "avgValue": 0.9},
            {"move": "L8", "status": "UNSOLVED", "avgValue": 0.1},
        ],
    }
    kv_store.save_entry(moves, result=result, db=db)  # L9/L8 have no entries of their own at all yet

    leaf = kv_store.find_deepest_promising_leaf(moves, db=db)

    assert leaf == ["K10", "L9"]  # the higher-value candidate - and the real next thing to search


def test_find_deepest_promising_leaf_chains_through_multiple_expanded_levels(db):
    moves = ["K10"]
    kv_store.save_entry(
        moves,
        result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "L9", "status": "UNSOLVED", "avgValue": 0.5}]},
        db=db,
    )
    kv_store.save_entry(
        moves + ["L9"],
        result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "M7", "status": "UNSOLVED", "avgValue": 0.5}]},
        db=db,
    )
    # M7 itself has no entry yet - the real frontier, two levels down.

    leaf = kv_store.find_deepest_promising_leaf(moves, db=db)

    assert leaf == ["K10", "L9", "M7"]


def test_find_deepest_promising_leaf_lands_on_a_proven_but_never_independently_searched_stub(db):
    """The self-healing case: a move the parent's own one-ply engine read
    already proved SOLVED_LOSS (resolving the parent to SOLVED_WIN via
    compute_book_solved_status's engine-status fallback) without that move
    ever being independently searched - still a bare add_parent_edge stub,
    no result of its own. find_deepest_promising_leaf should land exactly
    there, regardless of the parent already being solved - that's precisely
    how such a stub gets turned into real, independently-verified data."""
    moves = ["K10"]
    result = {"solvedStatus": "UNSOLVED", "topMoves": [{"move": "L9", "status": "SOLVED_LOSS", "avgValue": -1.0}]}
    hash_hex = kv_store.save_entry(moves, result=result, db=db)
    kv_store.add_parent_edge(moves + ["L9"], parent_hash=hash_hex, db=db)  # bare stub - no result
    kv_store.propagate_book_status(hash_hex, db=db)
    assert kv_store.get_entry(hash_hex, db=db)["bookSolvedStatus"] == "SOLVED_WIN"  # sanity: already "solved"

    leaf = kv_store.find_deepest_promising_leaf(moves, db=db)

    assert leaf == ["K10", "L9"]


def test_find_deepest_promising_leaf_stops_at_a_terminal_node_with_no_candidates(db):
    """A position that's been searched but found no (or no longer has any)
    candidate moves - compute_book_best_move returns (None, None) for it -
    is the end of the line, not an infinite loop."""
    moves = ["K10"]
    kv_store.save_entry(moves, result={"solvedStatus": "SOLVED_DRAW", "topMoves": []}, db=db)

    leaf = kv_store.find_deepest_promising_leaf(moves, db=db)

    assert leaf == moves


# ─── symmetry: canonical grouping shares proof across orientations ───────────


def test_get_group_status_finds_a_proven_twin(db):
    # ["A1", "B2"] and ["A19", "B18"] are genuine board-symmetric twins
    # (found empirically - see the API's git history for how) despite having
    # completely different move histories/parents.
    a_hash = kv_store.save_entry(["A1", "B2"], result={"solvedStatus": "SOLVED_WIN", "topMoves": []}, db=db)
    kv_store.propagate_book_status(a_hash, db=db)  # save_entry alone doesn't compute bookSolvedStatus
    canonical_hash, _ = kv_store.compute_canonical_hash(["A19", "B18"])

    assert kv_store.get_group_status(canonical_hash, db=db) == "SOLVED_WIN"


def test_get_group_status_excludes_the_given_hash(db):
    # Without exclude_hash, an entry reading its own group would just read
    # back its own (possibly stale, being-recomputed) status - see
    # compute_book_solved_status's use of this.
    hash_hex = kv_store.save_entry(["A1", "B2"], result={"solvedStatus": "SOLVED_WIN", "topMoves": []}, db=db)
    entry = kv_store.get_entry(hash_hex, db=db)

    assert kv_store.get_group_status(entry["canonicalHash"], db=db, exclude_hash=hash_hex) == "UNSOLVED"


def test_propagation_shares_proof_across_symmetric_twins_with_different_parents(db):
    """Parent1 -> child A. Parent2 -> child B. A and B are never played from
    the same parent, or from each other - just genuine board-symmetric twins
    of one another. Proving A should make Parent1 LOSS as usual, but also -
    without B ever itself being evaluated - make Parent2 LOSS too, since B
    reads as proven via A through their shared canonical group."""
    parent1_moves = ["A1"]
    parent2_moves = ["A19"]
    a_moves = ["A1", "B2"]
    b_moves = ["A19", "B18"]

    parent1_hash = kv_store.save_entry(
        parent1_moves, result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "B2"}]}, db=db
    )
    parent2_hash = kv_store.save_entry(
        parent2_moves, result={"solvedStatus": "UNSOLVED", "topMoves": [{"move": "B18"}]}, db=db
    )
    a_hash = kv_store.add_parent_edge(a_moves, parent_hash=parent1_hash, db=db)
    kv_store.add_parent_edge(b_moves, parent_hash=parent2_hash, db=db)  # B stays bare - never evaluated

    kv_store.save_entry(a_moves, result={"solvedStatus": "SOLVED_WIN", "topMoves": []}, db=db)
    kv_store.propagate_book_status(a_hash, db=db)

    assert kv_store.get_entry(parent1_hash, db=db)["bookSolvedStatus"] == "SOLVED_LOSS"
    assert kv_store.get_entry(parent2_hash, db=db)["bookSolvedStatus"] == "SOLVED_LOSS"
