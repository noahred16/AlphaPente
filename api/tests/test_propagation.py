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
        (["WIN"], "LOSS"),
        (["LOSS", "WIN", "UNSOLVED"], "LOSS"),  # one WIN is enough, regardless of the rest
        (["LOSS", "LOSS"], "WIN"),  # every reply bad for the opponent
        (["LOSS", "UNSOLVED"], "UNSOLVED"),  # unresolved blocks a conclusion even with no WIN
        (["LOSS", "DRAW"], "DRAW"),  # settled mix of LOSS/DRAW, no WIN, nothing pending
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
    kv_store.save_entry(a_moves, result={"solvedStatus": "LOSS", "topMoves": []}, db=db)
    kv_store.propagate_book_status(a_hash, db=db)
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"

    # Prove B as LOSS too - now every child of P is LOSS, so P proves WIN,
    # which in turn proves G LOSS (any WIN among G's children -> LOSS).
    b_hash = kv_store.add_parent_edge(b_moves, parent_hash=p_hash, db=db)
    kv_store.save_entry(b_moves, result={"solvedStatus": "LOSS", "topMoves": []}, db=db)
    kv_store.propagate_book_status(b_hash, db=db)
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "WIN"
    assert kv_store.get_entry(g_hash, db=db)["bookSolvedStatus"] == "LOSS"

    # Now add a brand-new, never-evaluated candidate "J10" to P's allowed set.
    kv_store.save_entry(p_moves, allowed_moves=["L9", "L8", "J10"], db=db)
    kv_store.propagate_book_status(p_hash, db=db)

    # P can no longer claim WIN - J10 is unresolved and might refute it - and
    # that reversion must cascade back up to G too, not just stop at P.
    assert kv_store.get_entry(p_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"
    assert kv_store.get_entry(g_hash, db=db)["bookSolvedStatus"] == "UNSOLVED"
