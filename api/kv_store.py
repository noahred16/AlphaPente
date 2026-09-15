import json

from rocksdict import Rdict

from api.config import settings
from api.zobrist import compute_hash

_book_db: Rdict | None = None


def get_book_db() -> Rdict:
    """Lazily open the real book_db (settings.book_db_path) on first use, not
    at import time - so importing this module (e.g. from tests, which inject
    their own temporary db) never touches the real on-disk database."""
    global _book_db
    if _book_db is None:
        _book_db = Rdict(settings.book_db_path)
    return _book_db


def save_result(moves: list[str], result: dict, db: Rdict | None = None) -> str:
    """Store a search result for the position reached by `moves`, keyed by
    its Zobrist hash. Returns the hash."""
    db = db if db is not None else get_book_db()
    hash_hex = compute_hash(moves)
    db[hash_hex] = json.dumps({"moves": moves, "result": result})
    return hash_hex


def get_result(hash_hex: str, db: Rdict | None = None) -> dict | None:
    """Look up a previously stored {"moves", "result"} entry by hash, or
    None if nothing is stored for it."""
    db = db if db is not None else get_book_db()
    raw = db.get(hash_hex)
    return json.loads(raw) if raw is not None else None
