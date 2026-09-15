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


def save_entry(moves: list[str], job_status: str, result: dict | None = None, db: Rdict | None = None) -> str:
    """Create or update the book entry for the position reached by `moves`,
    keyed by its Zobrist hash. Only the book_db owner (the Celery worker)
    should call this without an explicit db - see get_book_db(). `date_started`
    is set once (first write for a given hash) and preserved across later
    updates, e.g. when a queued job's result comes back. Returns the hash.
    """
    db = db if db is not None else get_book_db()
    hash_hex = compute_hash(moves)

    existing = get_entry(hash_hex, db=db)
    date_started = existing["date_started"] if existing else datetime.now(timezone.utc).isoformat()

    db[hash_hex] = json.dumps(
        {
            "moves": moves,
            "jobStatus": job_status,
            "date_started": date_started,
            "result": result,
        }
    )
    return hash_hex


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
