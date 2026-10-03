# Opening Book Builder — Architecture

This documents the web app for interactively building and curating a Pente
opening book: a FastAPI + Celery + RocksDB backend and a vanilla-JS frontend.

**This is a different kind of work from the rest of the repo.** The C++ MCTS
solver (`src/`, `include/`, `apps/` — see the root `CLAUDE.md`, `README.md`,
`ParallelDesign.md`, `Optimization.md`) is the actual game-solving engine,
built and optimized independently of this app. This app never reimplements
any of that — it shells out to the compiled `pente` binary and parses its
JSON output (`api/engine.py`). The solver has no idea this app exists; it's
just a subprocess as far as the engine is concerned.

## ⚠️ book_db is live and in active use

The opening book in book_db is real, accumulated work, not disposable test
data — treat every change near it accordingly:

- **Never make a breaking change to the entry schema** (renaming/removing a
  field, repurposing what a value means) without a backward-compatible read
  path for entries that predate the change. The existing pattern to follow:
  a new field is added as optional, `save_entry` preserves it unless
  explicitly set, and readers fall back to the old behavior when it's absent
  (see `bookBestMove`'s fallback to `result["bestMove"]` in
  `api/routers/book.py`). An old entry should keep working, not get
  silently corrupted or crash the API.
- **Never delete or bulk-modify existing data** — no migration/backfill
  scripts, no "fix up every record" passes, nothing that calls `destroy()`,
  `delete_range()`, or similar across the whole DB, even to retroactively
  apply a bug fix. Write the fix so it self-heals naturally the next time a
  position is touched through its normal path (`propagate_book_status` via
  `evaluate_position`/`set_allowed_moves`) instead — stale data left behind
  by a since-fixed bug is expected and fine; proactively rewriting it is not.
- **Verify against the real, live data before concluding something's
  broken** — a read-only `GET` (or a direct `rocksdict` secondary-mode
  query) is always safe and is how several real bugs in this system were
  actually found and confirmed, rather than guessed at.
- **Test schema/logic changes against a temporary RocksDB** (as the test
  suite already does, e.g. `api/tests/test_propagation.py`'s `db` fixture),
  never against the live book_db.
- Before any git operation that can discard uncommitted work (`checkout`,
  `restore`, `reset`, `clean`) on files in `api/`, check `git status` first
  — this exact mistake happened once already during this project's
  development and had to be manually reconstructed from scratch.

## Components at a glance

```
 docs/                    api/ (FastAPI)         api/tasks (Celery worker)
 vanilla JS/HTML/CSS  -->  HTTP endpoints    -->  evaluate_position
 served as static         dispatches tasks       set_allowed_moves
 files, NOT part          reads book_db           |
 of docker-compose        (RocksDB "secondary")    v
                                                 shells out to build/pente
                                                   |
                                                   v
                          book_db (RocksDB) <-- the ONLY read-write owner
                          Docker volume "book_data"
```

`redis` is Celery's broker + result backend. `docker-compose.yml` wires
`redis`, `api`, and `worker` together; `docs/` is plain static files and is
*not* one of the compose services — it's served independently.

## Backend

### Processes and why they're shaped this way

- **`api` service**: FastAPI (`api/main.py`). Handles HTTP, dispatches
  Celery tasks, and reads book_db via RocksDB's "secondary" instance mode —
  a read-only replica handle, safe to hold open in a different process from
  the writer. Never writes to book_db directly.
- **`worker` service**: a Celery worker (`api/tasks/book.py`) running
  `--pool=threads --concurrency=100`. The **sole** read-write owner of
  book_db.
- **`redis`**: broker + result backend.

Why `--pool=threads`, specifically (not the default prefork, not solo):
RocksDB allows only one read-write handle on a given path at a time —
prefork forks one subprocess per CPU core, and two of them independently
opening book_db collide ("IO error: ... LOCK: Resource temporarily
unavailable"). Threads keep everything in one process (one book_db handle)
but still run concurrently as OS threads. Solo (one task at a time, globally)
was tried first, but it meant a slow `evaluate_position` could starve a fast
`set_allowed_moves` sitting behind it in Celery's own queue, timing out the
PUT endpoint's synchronous `.get()`.

Running several tasks on real OS threads only works because RocksDB itself
is fine with concurrent access from multiple threads in one process, but the
read-modify-write JSON bookkeeping *around* it (`kv_store.py`'s functions all
do read → modify dict → write back) isn't — so every book_db write in
`kv_store.py` is wrapped in `_book_db_write_lock` (an `RLock`, not a plain
`Lock`, since functions like `propagate_book_status` recurse into themselves
and call each other while already holding it).

### Key files

| File | Role |
|---|---|
| `api/routers/book.py` | HTTP endpoints: `GET`/`POST /pente/book`, `PUT /pente/book/allowed-moves`, `GET /pente/book/queue`. Maps a raw book_db entry to the API's response schema (`_to_book_entry`), including several *derived* (not stored verbatim) fields — see below. |
| `api/kv_store.py` | All book_db read/write logic: entries, the parent DAG, canonical symmetry grouping, solved-status/best-move propagation, and the queue registry. |
| `api/tasks/book.py` | The two Celery tasks (`evaluate_position`, `set_allowed_moves`), the search-level tables, the single-search semaphore. |
| `api/engine.py` | Subprocess shim: runs `./pente <moves> <iterations> -n -j`, parses the JSON on stdout. |
| `api/zobrist.py` | `compute_hash` (position → key) and `compute_canonical_hash` (position → symmetry-group key), used to dedupe board-symmetric positions. |
| `api/schemas/book.py` | Pydantic request/response models. |
| `api/config.py` | Settings (`book_db_path`, `redis_url`) — `BOOK_DB_PATH=/data/book.db` is baked into the Dockerfile via `ENV`, overriding the plain-relative-path default. |

### book_db entry shape

One JSON blob per key, keyed by the position's Zobrist hash:

```jsonc
{
  "moves": ["K10", "L9", ...],
  "jobStatus": "IDLE" | "IN_PROGRESS" | "FAILED",   // "QUEUED" is an API-response-only state, see Queueing below
  "date_started": "...",                             // set once, first write, never overwritten
  "result": { ... } | null,                          // the engine's raw JSON output, verbatim, or null if never evaluated
  "allowedMoves": ["L9", ...] | null,                // null = "nothing restricted yet, allow everything"
  "targetVisits": 200000 | null,                     // the iteration count last *requested* - not necessarily what ran, a proof can finish early
  "parentHashes": ["<hash>", ...],
  "bookSolvedStatus": "UNSOLVED" | "SOLVED_WIN" | "SOLVED_LOSS" | "SOLVED_DRAW",
  "bookBestMove": "K13" | null,
  "bookBestValue": 0.42 | null,
  "canonicalHash": "<hash>",
  "sym": 4
}
```

Three disjoint key namespaces share one RocksDB keyspace, never colliding:

- a plain hex hash → a position entry (above)
- `canon:<hash>` → the list of physical hashes belonging to one symmetry group
- `queue:jobs` → the single queue-registry document (see Queueing below)

### Solved-status / best-move propagation

Every position records its own `parentHashes` via `add_parent_edge` — every
position known to reach it as a candidate child. This forms a DAG (provably
acyclic: capture counts baked into the hash never decrease, and every move
adds exactly one stone, so it can't loop back on itself).

`propagate_book_status(hash)` recomputes **both** `bookSolvedStatus` and
`bookBestMove`/`bookBestValue` for one position and, only if *either*
changed, recurses into every recorded parent (and every symmetric twin's
parents). It's called at the end of both `evaluate_position` and
`set_allowed_moves` — the only two places book_db is ever written to, and so
the only two places a change could need to ripple upward.

- `compute_book_solved_status`: negamax over candidate children's *freshest
  known* status — a child's own independently-computed book-level status
  (via its symmetry group, so a proof found through a board-symmetric twin
  counts too) if it has one, **else this search's own one-ply engine read on
  it**. That fallback matters even for a child that's never been
  independently touched at all: a single search can fully prove several of
  its own replies internally (as part of one long run) without any of them
  ever becoming their own independently-searched position, and without the
  fallback, that proof was invisible to `compute_book_solved_status` — the
  position just stayed `UNSOLVED` forever despite every reply already being
  a proven loss. (This was a real, fixed bug — see git history around
  `compute_book_solved_status`.)
- `compute_book_best_move`: same "freshest known" idea, ranking candidates
  by (proven win > not-a-proven-loss > proven loss), then by the deepest
  known value — a child's own live best value if it's been computed, else
  its one-ply engine read. The ranking mirrors `docs/js/book.js`'s
  `compareByValue` exactly, so the table's own #1 row always agrees with
  what the backend considers best.
- Canonical symmetry grouping (`add_canonical_member` / `get_group_status`)
  is what lets a proof found via *one* board orientation count for every
  rotation/reflection of it, without re-searching them.

### The sign convention (confirmed empirically, easy to get backwards)

Every node's own value/status is recorded **from the perspective of whoever
just moved into it** (the "last mover"), not whoever's about to move there.
This was confirmed by hand-building a forced-loss position (an unstoppable
open four) and checking that the engine reports the *losing* side's own
replies as `SOLVED_LOSS` with a very negative value.

Practical effect: a candidate move's own `avgValue`/`status` are *already*
in the perspective of the player choosing among them — no flip needed to
display them. But drilling one ply further (a move's own child's best
reply) needs exactly one flip. The root-level `bestValue`/`rootAvgValue`
use the *opposite* convention (whoever just moved into *this* position) for
consistency with the engine's own raw output — so `compute_book_best_move`
has to flip its internal (correctly-signed-for-*ranking*) value before
returning it, specifically to keep agreeing with that older convention.

## The queueing system

- **Only one real MCTS search ever runs at a time**, globally, via
  `_search_semaphore` (a plain `threading.Semaphore(1)`) in
  `api/tasks/book.py` — regardless of how many `evaluate_position` calls are
  dispatched. Each search assumes it owns the whole machine
  (`NUM_THREADS` worker threads, `ARENA_SIZE_GB` memory), so running more
  than one concurrently thrashes the system.
- A backlogged `evaluate_position` just blocks on `.acquire()` — it does
  **not** retry or reschedule itself. A blocked thread costs nothing (no
  CPU, just a sleeping OS thread), but it *does* occupy a thread-pool slot
  — which is why `--concurrency` matters (currently 100, raised from 32
  after a real backlog of exactly 32 queued positions hit that ceiling and
  started starving `set_allowed_moves` of a free thread to run on at all).
- The `"QUEUED"` job status a user sees is **not** written to book_db the
  moment a job is dispatched — `evaluate_position` writes it as its own
  first real action, before even touching the semaphore. (This, too, was a
  fixed bug: previously nothing was written until the job actually started
  running, so a position stuck waiting in a real backlog looked — to every
  subsequent `GET` poll — like it had never been queued at all, and got
  re-dispatched again and again, each duplicate making the real backlog
  longer.)
- `register_queued_job` / `unregister_queued_job` (`kv_store.py`) maintain a
  separate, explicit registry: the single `queue:jobs` document lists every
  `evaluate_position` call dispatched but not yet finished — including the
  one actually running (there's no reliable way to tell which registry
  entry that is). This is exactly what `GET /pente/book/queue` reports and
  the frontend's Queue panel renders, entirely independent of asking
  Celery/Redis anything directly.
- The registry resets to empty when the worker process starts
  (`worker_ready` Celery signal) — anything recorded as in-flight at that
  moment didn't survive whatever just (re)started the process.
- **Search levels** (`SearchLevel` enum; `SEARCH_LEVEL_ITERATIONS` /
  `SEARCH_LEVEL_SECONDS` in `api/tasks/book.py`), each independently
  calibrated against this machine's config (per-iteration throughput isn't
  linear — thread/tree-setup overhead dominates at low counts):

  | Level | Iterations | Est. time |
  |---|---|---|
  | VERY_FAST | 200,000 | ~5s |
  | FAST | 2,000,000 | ~30s |
  | MEDIUM | 9,000,000 | ~2min |
  | DEEP | 900,000,000 | ~6min ballpark (really arena-bound) |
  | MAX | 999,000,000 | none — "whatever this machine can hold" |

## Frontend (`docs/`)

- `docs/index.html` + `docs/js/book.js` + `docs/css/style.css`: one
  single-page vanilla-JS app, no build step, no framework. Plain static
  files, served independently of docker-compose.
- Uses the same WASM `PenteGame` module as the 5x5 demo page, purely for
  **local** rule enforcement (whose turn, legal moves, captures, 5-in-a-row)
  — all actual book data comes from the API.
- URL state: `?moves=K10,L9,...` round-trips the current position via
  `history.replaceState` — shareable, bookmarkable links, used throughout
  (the Queue panel's and Saved Games panel's own move links are plain
  `<a href="?moves=...">` tags).
- localStorage-held UI state (all best-effort, wrapped in try/catch for
  private browsing): `queuedMoves` / `deepeningMoves` (bridges the gap
  between clicking Queue/Deepen and the backend confirming it — real
  backend signals like `childInProgress`/`expanded` take over once
  registered; these flags carry a 5-minute TTL as a self-healing safety net
  against ever getting permanently stuck), the selected search level, and
  starred/saved positions (deliberately *not* in book_db — a per-browser
  bookmark is not book data).
- The moves table's row order and the board's ghost-overlay numbering both
  use `compareByValue`, which mirrors `compute_book_best_move`'s backend
  ranking exactly (status first, then Child Val, then MCTS Val) so the
  table's own #1 row never disagrees with what the backend considers best.
- The Queue panel polls `GET /pente/book/queue` on its own timer,
  independent of the per-position poll loop, and chains each job's own
  estimated duration into a running ETA schedule.

## Known gotchas worth remembering

- **RocksDB single-writer.** Only the worker ever calls `get_book_db()`
  (primary, read-write). Everything else uses `get_book_db_reader()`
  (secondary, read-only, calling `try_catch_up_with_primary()` on every use
  so it never serves a stale snapshot).
- **book_db does not automatically survive an ungraceful worker restart**
  while jobs are in flight. Celery acks a task the moment a thread picks it
  up, not when it finishes — a dequeued-but-not-yet-run task is lost
  outright if the worker dies before finishing it, not automatically
  retried. Check `GET /pente/book/queue`'s live count before restarting the
  worker; a nonzero count means re-queuing some positions by hand afterward.
- **Backup/restore**: stop `api` + `worker`, tar the `book_data` Docker
  volume's contents, restart. `rocksdict` exposes no checkpoint API, so
  copying it live risks an inconsistent snapshot.
