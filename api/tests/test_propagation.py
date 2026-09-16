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
