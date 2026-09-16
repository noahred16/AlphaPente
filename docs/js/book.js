// 19x19 Pente opening-book builder. The WASM PenteGame (same module as the
// 5x5 demo) enforces the actual rules locally - alternating turns, captures,
// 5-in-a-row - while the backend book API (api/routers/book.py) supplies
// candidate-move stats for the current position and owns queuing MCTS
// searches into book_db.
const BOARD_SIZE = 19;
const CENTER = Math.floor(BOARD_SIZE / 2); // 9 -> "K10", Pente's forced first move
// Same host the page itself was loaded from (not a hardcoded "localhost") so
// this still reaches the API when the frontend is opened from another
// machine, e.g. over Tailscale - only the port differs from the page's own.
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
const POLL_INTERVAL_MS = 15000;

const QUEUED_STORAGE_KEY = 'pente-book-queued-moves';
const DEEPENING_STORAGE_KEY = 'pente-book-deepening-moves';
const LEVEL_STORAGE_KEY = 'pente-book-search-level';

// Mirrors api/tasks/book.py's SEARCH_LEVEL_ITERATIONS exactly (same hardcoded
// values, same reasoning) - every targetVisits/childTargetVisits the backend
// ever hands back is exactly one of these three numbers (or null, for a
// position that's never been evaluated), so labeling one back into "Fast"/
// "Medium"/"Deep" (see levelLabel) is a plain reverse lookup, not a guess.
const SEARCH_LEVEL_ITERATIONS = { VERY_FAST: 200_000, FAST: 2_000_000, MEDIUM: 9_000_000, DEEP: 900_000_000 };

let Module, game;
let moveHistory = [];  // move strings in play order, e.g. ["K10", "L9"]; always starts with the forced center opening
let redoStack = [];    // move strings popped off by undo, replayed by redo
let bookEntry = null;  // last GET /pente/book response for the current position, or null while loading
let queuedMoves = loadQueuedMoves(); // "movesJSON|move" keys the user has queued, persisted across reloads - see queueKey()
let deepeningMoves = loadDeepeningMoves(); // same shape, for moves (or the root - see rootDeepenKey) requeued at a deeper level
let requestToken = 0;  // guards against a stale fetch response overwriting a newer one
let pollTimer = null;
let allowedMovesBusy = false; // guards against overlapping PUT /allowed-moves calls - see addAllowedMove/removeAllowedMove

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const capturesEl = document.getElementById('captures');
const movesBody = document.querySelector('#moves-table tbody');
const resetBtn = document.getElementById('reset');
const undoBtn = document.getElementById('undo');
const redoBtn = document.getElementById('redo');
const refreshBtn = document.getElementById('refresh');
const levelSelect = document.getElementById('level');
const queueCountEl = document.getElementById('queue-count');
const queueSummaryEl = document.getElementById('queue-summary');
const queueTableBody = document.querySelector('#queue-table tbody');

// The search level applied to every evaluation this tab queues from now on
// (GET's auto-queue of a freshly-seen position, and the table's Queue/Deepen
// buttons) - persisted so it survives a reload, same pattern as queuedMoves.
try {
  const savedLevel = localStorage.getItem(LEVEL_STORAGE_KEY);
  if (savedLevel && SEARCH_LEVEL_ITERATIONS[savedLevel]) levelSelect.value = savedLevel;
} catch (e) {
  // ignore - purely a convenience, not a source of truth
}
levelSelect.addEventListener('change', () => {
  try {
    localStorage.setItem(LEVEL_STORAGE_KEY, levelSelect.value);
  } catch (e) {
    // ignore
  }
  render(); // a Queue button may need to become a Deepen button (or vice versa) at the new level
});

function currentLevel() {
  return levelSelect.value;
}

// Reverse-lookup a stored targetVisits back into its level name for display -
// null for a position that's never been evaluated, or (in principle, for
// data predating this feature) a targetVisits that doesn't match any level.
function levelLabel(targetVisits) {
  return Object.keys(SEARCH_LEVEL_ITERATIONS).find(level => SEARCH_LEVEL_ITERATIONS[level] === targetVisits) || null;
}

// Whether the currently-selected level would search deeper than a position's
// last recorded targetVisits - the condition for offering a "Deepen" requeue
// rather than just showing what's already there. A position that's never
// been evaluated (targetVisits null) isn't "deepenable" - it just needs a
// plain Queue instead.
function isDeeper(targetVisits) {
  return targetVisits != null && SEARCH_LEVEL_ITERATIONS[currentLevel()] > targetVisits;
}

// Whether the currently-selected level is exactly what a position was last
// run at - the condition for offering a plain "Re-run" (e.g. to get a fresh
// result at the same depth) rather than a "Deepen".
function isSameLevel(targetVisits) {
  return targetVisits != null && SEARCH_LEVEL_ITERATIONS[currentLevel()] === targetVisits;
}

// The requeue button label to offer for a position last run at `targetVisits`,
// or null if the selected level offers nothing beyond what's already there.
function rerunLabel(targetVisits) {
  if (isDeeper(targetVisits)) return 'Deepen';
  if (isSameLevel(targetVisits)) return 'Re-run';
  return null;
}

// Real Pente: the engine's BLACK always moves first (see PenteGame::reset) -
// but this app displays the first mover as white stones instead (and the
// second mover as black), so every display of a stone/player color goes
// through here rather than assuming engine-BLACK-looks-black.
function displayColor(player) {
  return player === 1 ? 'White' : 'Black';
}

// --- Move <-> board coordinate conversion (mirrors GameUtils::parseMove/
// displayMove: column letters A-T skip 'I', rows are 1-indexed) ---
function colLetter(x) {
  let code = 'A'.charCodeAt(0) + x;
  if (code >= 'I'.charCodeAt(0)) code++;
  return String.fromCharCode(code);
}

function moveStr(x, y) {
  return colLetter(x) + (y + 1);
}

function parseMoveStr(move) {
  let code = move.charCodeAt(0);
  if (code >= 'I'.charCodeAt(0)) code--;
  return { x: code - 'A'.charCodeAt(0), y: parseInt(move.slice(1), 10) - 1 };
}

// The board <div> for a move string - used to highlight a table row's move
// on the board while hovering its link (see renderMovesTable).
function cellAt(move) {
  const { x, y } = parseMoveStr(move);
  return boardEl.children[y * BOARD_SIZE + x];
}

// Same shape the backend enforces (api/schemas/book.py's MoveStr) - guards
// against feeding a malformed move (e.g. a hand-edited URL) into game.makeMove,
// which - like the native engine behind it - doesn't itself validate input.
const MOVE_PATTERN = /^[A-HJ-T](?:[1-9]|1[0-9])$/;

// --- URL <-> game state, so a refresh (or a shared link) resumes the same
// position instead of always starting over from the forced center opening. ---
function loadMovesFromUrl() {
  const raw = new URLSearchParams(location.search).get('moves');
  return raw ? raw.split(',').filter(m => MOVE_PATTERN.test(m)) : [];
}

function syncUrl() {
  const params = new URLSearchParams();
  if (moveHistory.length) params.set('moves', moveHistory.join(','));
  // replaceState, not pushState: every move already has its own undo/redo
  // entry in-app (see moveHistory/redoStack) - there's no need for the
  // browser's own back/forward to also step through each one.
  history.replaceState(null, '', moveHistory.length ? `?${params}` : location.pathname);
}

// --- Backend book API ---
async function fetchBookEntry(moves) {
  const params = new URLSearchParams();
  moves.forEach(m => params.append('moves', m));
  // Only takes effect the first time this exact position is seen (see
  // get_book_entry's docstring) - re-evaluating an already-searched position
  // deeper is queueEvaluation's job, not GET's.
  params.set('level', currentLevel());
  const res = await fetch(`${API_BASE}/pente/book?${params}`);
  if (!res.ok) throw new Error(`GET /pente/book failed: ${res.status}`);
  return res.json();
}

async function queueEvaluation(moves) {
  const res = await fetch(`${API_BASE}/pente/book`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves, level: currentLevel() }),
  });
  if (!res.ok) throw new Error(`POST /pente/book failed: ${res.status}`);
  return res.json();
}

async function putAllowedMoves(moves, allowedMoves) {
  const res = await fetch(`${API_BASE}/pente/book/allowed-moves`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves, allowedMoves }),
  });
  if (!res.ok) throw new Error(`PUT /pente/book/allowed-moves failed: ${res.status}`);
  return res.json();
}

// --- Queue panel: GET /pente/book/queue is global (every evaluate_position
// job dispatched but not yet finished, across every position - see
// api/routers/book.py's get_queue), unlike the rest of this file's fetches,
// which are all scoped to whatever position is currently open - so it's
// polled on its own timer instead of piggybacking on loadBookEntry/
// startPolling. Best-effort: a hiccup here shouldn't disturb the rest of the
// page, so failures are swallowed rather than surfaced via statusEl.
async function loadQueue() {
  try {
    const res = await fetch(`${API_BASE}/pente/book/queue`);
    if (!res.ok) throw new Error(`GET /pente/book/queue failed: ${res.status}`);
    renderQueue(await res.json());
  } catch (e) {
    // ignore - the next poll will try again
  }
}

// mm:ss-free, human-scale duration: "2h 15m", "15m", or "<1m" for something
// that rounds down to nothing.
function formatDuration(seconds) {
  const totalMinutes = Math.round(seconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m`;
  return '<1m';
}

function formatClockTime(date) {
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

// Chains each job's own estimatedSeconds (see QueuedJob) into a running
// schedule, in queue order: the first job is assumed to be the one actually
// searching right now (there's no reliable way to tell which registry entry
// that really is - see kv_store.register_queued_job - so "starts now" is the
// best available approximation), and each later job starts exactly when the
// one ahead of it ends. A DEEP job's estimatedSeconds is null (arena-bound,
// not time-bound - see SEARCH_LEVEL_SECONDS) - once one of those is hit, its
// own end and every job scheduled after it become unknown too, not just that
// one job.
function scheduleQueue(jobs) {
  let cursor = 0; // seconds from now, or null once unknown
  return jobs.map(job => {
    const startSeconds = cursor;
    const endSeconds = cursor == null || job.estimatedSeconds == null ? null : cursor + job.estimatedSeconds;
    cursor = endSeconds;
    return { ...job, startSeconds, endSeconds };
  });
}

function renderQueue({ count, jobs }) {
  queueCountEl.hidden = count === 0;
  queueCountEl.textContent = `Queue: ${count}`;

  const scheduled = scheduleQueue(jobs);
  const totalSeconds = scheduled.length ? scheduled[scheduled.length - 1].endSeconds : 0;
  if (jobs.length === 0) {
    queueSummaryEl.textContent = 'Queue is empty.';
  } else if (totalSeconds == null) {
    queueSummaryEl.textContent = `${count} job${count === 1 ? '' : 's'} queued - a Deep search makes the total time unknown.`;
  } else {
    const emptyAt = formatClockTime(new Date(Date.now() + totalSeconds * 1000));
    queueSummaryEl.textContent =
      `${count} job${count === 1 ? '' : 's'} queued - about ${formatDuration(totalSeconds)} total, empty around ${emptyAt}.`;
  }

  queueTableBody.innerHTML = '';
  scheduled.forEach((job, i) => {
    const row = document.createElement('tr');
    const level = levelLabel(job.targetVisits) || `${job.targetVisits}`;
    const starts = job.startSeconds == null ? '?' : job.startSeconds === 0 ? 'now' : formatClockTime(new Date(Date.now() + job.startSeconds * 1000));
    const ends = job.endSeconds == null ? '?' : formatClockTime(new Date(Date.now() + job.endSeconds * 1000));
    // Same query-string shape syncUrl() writes, so this is a plain link to
    // that exact game - not intercepted with playMove/onCellClick's usual
    // in-app handling, since it's a different position than whatever's open
    // right now; a real navigation just reloads the page onto it.
    const params = new URLSearchParams();
    params.set('moves', job.moves.join(','));
    row.innerHTML = `<td>${i + 1}</td><td><a class="move-link" href="?${params}">${job.moves.join(' ')}</a></td><td>${level}</td><td>${starts}</td><td>${ends}</td>`;
    queueTableBody.appendChild(row);
  });
}

// Key identifying "this candidate move, from this position" - used to
// remember which moves the user has queued (see renderMovesTable).
function queueKey(move) {
  return JSON.stringify(moveHistory) + '|' + move;
}

// Queued moves persist in localStorage (not just in memory) so a page reload
// doesn't forget a move was queued and offer to queue it again - useful since
// a real search can take a long time to actually start (see
// SEARCH_LEVEL_ITERATIONS in api/tasks/book.py) or finish. Falls back to an
// empty set if storage is unavailable (e.g. private browsing).
function loadQueuedMoves() {
  try {
    return new Set(JSON.parse(localStorage.getItem(QUEUED_STORAGE_KEY)) || []);
  } catch (e) {
    return new Set();
  }
}

function saveQueuedMoves() {
  try {
    localStorage.setItem(QUEUED_STORAGE_KEY, JSON.stringify([...queuedMoves]));
  } catch (e) {
    // ignore - purely a convenience, not a source of truth
  }
}

// Same idea as queueKey/queuedMoves above, but for the CURRENT position
// itself rather than one of its candidate moves - "root" can never collide
// with a real move string, so this is always distinguishable from a
// queueKey() result.
function rootDeepenKey() {
  return JSON.stringify(moveHistory) + '|root';
}

// Moves (or the current position - see rootDeepenKey) that have been
// requeued at a deeper level than they already have a result for (see
// isDeeper). Tracked separately from queuedMoves: unlike a fresh Queue,
// deepening starts from a position that's already "expanded" (see
// moveState), so the locally-known "there's a job in flight" state needs to
// live here to keep showing that instead of a Deepen button the user could
// otherwise click again before the new result lands.
function loadDeepeningMoves() {
  try {
    return new Set(JSON.parse(localStorage.getItem(DEEPENING_STORAGE_KEY)) || []);
  } catch (e) {
    return new Set();
  }
}

function saveDeepeningMoves() {
  try {
    localStorage.setItem(DEEPENING_STORAGE_KEY, JSON.stringify([...deepeningMoves]));
  } catch (e) {
    // ignore - purely a convenience, not a source of truth
  }
}

// Where a candidate move actually stands. `m.expanded` comes straight off
// the single GET /pente/book response - the backend computes it by looking
// up each move's own child position directly in book_db (see
// _to_book_entry's expanded_state in api/routers/book.py), so this needs no
// extra requests of its own. The locally-queued flag still fills one real
// gap in that: book_db itself never records a "QUEUED" jobStatus (only the
// Celery worker writes IN_PROGRESS, once it actually starts the job) - so a
// move can sit queued-but-not-yet-started for a while, especially behind a
// busy worker, with nothing in book_db yet to show for it.
function moveState(m) {
  if (m.expanded === 'true') return 'expanded';
  if (m.expanded === 'in progress') return 'in-progress';
  if (queuedMoves.has(queueKey(m.move))) return 'queued';
  return 'none';
}

// --- Game setup ---
// `initialMoves` (from the URL - see loadMovesFromUrl) is replayed instead of
// just the forced center opening, so a refresh or a shared link resumes the
// same position. Stops at (and drops) the first illegal move it hits - a
// stale or hand-edited URL - falling back to the plain forced opening if
// nothing in it was even replayable.
function newGame(initialMoves = []) {
  if (game) game.delete();
  game = new Module.Game(BOARD_SIZE, 1); // simulations param is unused - this Game is only used here for rule enforcement
  moveHistory = [];
  redoStack = [];
  for (const m of initialMoves) {
    const { x, y } = parseMoveStr(m);
    if (!game.makeMove(x, y)) break;
    moveHistory.push(m);
  }
  if (moveHistory.length === 0) {
    game.makeMove(CENTER, CENTER);
    moveHistory.push(moveStr(CENTER, CENTER));
  }
  buildBoard();
  loadBookEntry();
}

function buildBoard() {
  boardEl.innerHTML = '';
  boardEl.style.setProperty('--board-size', BOARD_SIZE);
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.tooltip = moveStr(x, y); // CSS tooltip (see style.css) - which square this is
      cell.addEventListener('click', () => onCellClick(x, y));
      boardEl.appendChild(cell);
    }
  }
}

function replayHistory() {
  game.reset();
  for (const m of moveHistory) {
    const { x, y } = parseMoveStr(m);
    game.makeMove(x, y);
  }
}

// --- Rendering ---
function render() {
  renderBoard();
  renderMovesTable();
  renderStatus();
}

function renderBoard() {
  const cells = boardEl.children;
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      const cell = cells[y * BOARD_SIZE + x];
      cell.innerHTML = '';
      cell.classList.remove('last-move', 'hovered');
      cell.style.boxShadow = '';
      cell.dataset.tooltip = moveStr(x, y); // plain coordinate by default - see the ghost loop below for Child Val
      const stoneVal = game.getStoneAt(x, y); // 0=empty, 1=engine BLACK, 2=engine WHITE
      cell.classList.toggle('occupied', stoneVal !== 0);
      if (stoneVal === 1 || stoneVal === 2) {
        const stone = document.createElement('div');
        stone.className = 'stone ' + displayColor(stoneVal).toLowerCase();
        cell.appendChild(stone);
      }
    }
  }

  const last = moveHistory[moveHistory.length - 1];
  if (last) {
    const { x, y } = parseMoveStr(last);
    cells[y * BOARD_SIZE + x].classList.add('last-move');
  }

  // Ghost overlay: every currently-allowed candidate reply the book knows
  // about for the side to move, numbered by rank - a removed move doesn't
  // linger here dimmed, since removeAllowedMove really deletes its data now
  // (see allowedTopMoves()), not just detaches it from view. Not-yet-queued
  // moves are faint; queued/in-progress/expanded ones are shown solid but
  // still lighter than a real stone.
  if (bookEntry) {
    const toMoveColor = displayColor(game.getCurrentPlayer()).toLowerCase();
    const moves = allowedTopMoves();
    // Priors come from the engine's own heuristic policy (see
    // HeuristicEvaluator::evaluatePolicy / PenteGame::evaluateMove), but
    // every "boring" candidate move already gets a flat nonzero baseline
    // score before any bonus for an actual pattern (open three, capture
    // threat/defense, four threat, etc.) is added - so most moves share
    // that same baseline prior, and "prior > 0" alone doesn't mean
    // tactically relevant. Find that baseline (the most common prior here,
    // same idea as ParallelMCTS::getTopMoves' server-side cutoff) and only
    // darken moves that actually rise above it, scaled by how far above.
    const baseline = modePrior(moves.map(m => m.prior));
    const maxPrior = Math.max(0, ...moves.map(m => m.prior));
    moves.forEach((m, i) => {
      const { x, y } = parseMoveStr(m.move);
      const cell = cells[y * BOARD_SIZE + x];
      if (cell.classList.contains('occupied')) return; // shouldn't happen, but never draw a ghost over a real stone
      if (m.prior > baseline * 1.0001 && maxPrior > baseline) {
        const darkness = 0.35 * (m.prior - baseline) / (maxPrior - baseline);
        cell.style.boxShadow = `inset 0 0 0 999px rgba(0,0,0,${darkness})`;
      }
      if (m.childBestMoveValue != null) {
        cell.dataset.tooltip = `${m.move} - ${m.childBestMoveValue.toFixed(3)}`;
      }
      const expanded = moveState(m) !== 'none';
      const ghost = document.createElement('div');
      ghost.className = 'stone ghost ' + toMoveColor + (expanded ? ' expanded' : '');
      ghost.textContent = String(i + 1);
      cell.appendChild(ghost);
    });
  }
}

// The most common prior among a list (rounded to tolerate float noise) -
// see renderBoard's use of this as the "ordinary move" baseline to darken
// squares against.
function modePrior(priors) {
  const counts = new Map();
  let best = 0, bestCount = 0;
  for (const p of priors) {
    if (p <= 0) continue;
    const key = Math.round(p * 1e6);
    const count = (counts.get(key) || 0) + 1;
    counts.set(key, count);
    if (count > bestCount) {
      bestCount = count;
      best = p;
    }
  }
  return best;
}

// Ranks two moves by how good they look, best first: Child Val (the deeper,
// more authoritative number - see TopMove.childBestMoveValue) if both have
// one, falling back to MCTS Val (avgValue, flipped - see renderMovesTable)
// when either doesn't. A move with no numeric read at all on the field being
// compared sorts after one that has it; Array.prototype.sort is stable, so
// two moves with nothing to compare keep the engine's own strength order
// (see ParallelMCTS::getTopMoves) instead of getting shuffled arbitrarily.
function compareByValue(a, b) {
  if (a.childBestMoveValue != null && b.childBestMoveValue != null) {
    return b.childBestMoveValue - a.childBestMoveValue;
  }
  if (a.childBestMoveValue != null) return -1;
  if (b.childBestMoveValue != null) return 1;

  const aValue = a.avgValue != null ? -a.avgValue : null;
  const bValue = b.avgValue != null ? -b.avgValue : null;
  if (aValue != null && bValue != null) return bValue - aValue;
  if (aValue != null) return -1;
  if (bValue != null) return 1;
  return 0;
}

// The subset actually in the position's allowedMoves right now, best-ranked
// first (see compareByValue) - what the board's ghost overlay (see
// renderBoard) and moves table (see renderMovesTable) both show and number,
// and the base for building a new list to PUT (see
// addAllowedMove/removeAllowedMove). A move the engine ranked in its own
// topMoves but that got removed here doesn't linger anywhere once
// removeAllowedMove deletes its underlying data - see set_allowed_moves.
function allowedTopMoves() {
  return bookEntry ? bookEntry.topMoves.filter(m => m.isAllowed).sort(compareByValue) : [];
}

function allowedMoveList() {
  return allowedTopMoves().map(m => m.move);
}

function renderMovesTable() {
  movesBody.innerHTML = '';
  if (!bookEntry) return;
  // Only already-allowed moves - unlike the board's ghost overlay (see
  // renderBoard), which still shows every candidate the book knows about,
  // this table is specifically the allowed-move list: every row here is
  // already "in", so they render identically (no dimming) and only ever
  // offer removal, never adding - a not-yet-allowed move gets added by
  // clicking it on the board instead (see onCellClick).
  allowedTopMoves().forEach((m, i) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${i + 1}</td><td><a href="#" class="move-link">${m.move}</a></td><td></td><td>${m.childMoveCount}</td><td></td><td></td><td></td><td></td>`;
    const [priorCell, statusCell, valueCell, childValCell, actionsCell] =
      [row.children[2], row.children[4], row.children[5], row.children[6], row.children[7]];

    // Prior comes from the engine's own policy - null (well, 0, its
    // placeholder - see _to_book_entry) only for a manually-added move it
    // never itself evaluated, same condition as avgValue being null.
    priorCell.textContent = m.avgValue === null ? 'N/A' : m.prior.toFixed(3);

    const moveLink = row.querySelector('.move-link');
    // Same as clicking this move's cell on the board (see onCellClick) -
    // every row here is already allowed, so this always plays it.
    moveLink.addEventListener('click', e => {
      e.preventDefault();
      playMove(m.move);
    });
    // Highlight the corresponding board cell while hovering this link, so
    // it's easy to see where a move actually is without hunting for its
    // ghost overlay number.
    moveLink.addEventListener('mouseenter', () => cellAt(m.move).classList.add('hovered'));
    moveLink.addEventListener('mouseleave', () => cellAt(m.move).classList.remove('hovered'));

    const state = moveState(m);
    const pending = state === 'queued' || state === 'in-progress';

    // Status: a proven result beats everything else; a job in flight (either
    // backend-confirmed "in progress" or just locally queued - see
    // moveState) hasn't produced one yet, so that beats reporting how much
    // search this move already has; otherwise, once actually expanded, how
    // far it's been searched (in millions of visits - a plain read of the
    // level it was last run at, see SEARCH_LEVEL_ITERATIONS). Never
    // expanded at all: nothing to report yet.
    if (m.status !== 'UNSOLVED') {
      statusCell.textContent = m.status;
    } else if (pending) {
      statusCell.textContent = 'QUEUED';
    } else if (state === 'expanded' && m.childTargetVisits != null) {
      statusCell.textContent = `${+(m.childTargetVisits / 1e6).toFixed(2)}M`;
    }

    // MCTS Val: avgValue is stored from the perspective of the opponent (the
    // player to move at the resulting child position - see
    // ParallelMCTS::backpropagate's per-ply sign flip), so flip it back to
    // the perspective of the player actually choosing among these moves.
    // Null only for a manually-added move with no engine data yet (see
    // _to_book_entry) - nothing to flip there either.
    if (!pending) {
      valueCell.textContent = m.avgValue === null ? 'N/A' : (-m.avgValue).toFixed(3);
    }

    // Child Val: the avgValue of the best reply found by the child's own,
    // independent search - one ply deeper than MCTS Val, and (unlike MCTS
    // Val) already in this table's own to-move player's perspective, so no
    // flip here - see TopMove.childBestMoveValue. Blank if the child's own
    // search hasn't produced anything yet, same as Status's "never
    // expanded" case.
    if (m.childBestMoveValue != null) {
      childValCell.textContent = m.childBestMoveValue.toFixed(3);
    }

    // Actions: a Queue/Deepen/Re-run request (only ever one of the three -
    // see rerunLabel - nothing while a job's already in flight, or once
    // this move is solved, since a proven result doesn't need re-searching)
    // alongside the always-available remove control.
    if (!pending && m.status === 'UNSOLVED') {
      if (state === 'expanded') {
        // Suppressed once already requeued (see deepeningMoves) - or once
        // the worker's actually picked it up (childInProgress, the raw
        // signal `expanded` itself deliberately masks - see
        // expanded_state's docstring) - so it can't be double-clicked while
        // the new result is still in flight.
        if (deepeningMoves.has(queueKey(m.move)) || m.childInProgress) {
          actionsCell.appendChild(document.createTextNode('requeued… '));
        } else {
          const label = rerunLabel(m.childTargetVisits);
          if (label) {
            const btn = document.createElement('button');
            btn.className = 'queue-btn';
            btn.textContent = label;
            btn.addEventListener('click', () => onDeepenClick(m.move, btn, label));
            actionsCell.appendChild(btn);
            actionsCell.appendChild(document.createTextNode(' '));
          }
        }
      } else {
        const btn = document.createElement('button');
        btn.className = 'queue-btn';
        btn.textContent = 'Queue';
        btn.addEventListener('click', () => onQueueClick(m.move, btn));
        actionsCell.appendChild(btn);
        actionsCell.appendChild(document.createTextNode(' '));
      }
    }

    // Always a remove control - every row here is already allowed (see
    // allowedTopMoves()), so there's never an add case to offer.
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.title = `Delete ${m.move} (and its subtree, unless reachable elsewhere)`;
    removeBtn.textContent = '🗑';
    removeBtn.addEventListener('click', () => removeAllowedMove(m.move, removeBtn));
    actionsCell.appendChild(removeBtn);

    movesBody.appendChild(row);
  });
}

function renderStatus() {
  // getBlackCaptures/getWhiteCaptures are the engine's own (BLACK-moves-first)
  // color labels - see displayColor - so the label swap has to happen here too.
  capturesEl.textContent = `Captures — ${displayColor(1)}: ${game.getBlackCaptures()}, ${displayColor(2)}: ${game.getWhiteCaptures()}`;

  // Only worth reporting the book-level solved status once it's actually
  // settled - UNSOLVED is the default/common case, not news.
  const solvedSuffix = bookEntry && bookEntry.solvedStatus !== 'UNSOLVED' ? `, ${bookEntry.solvedStatus}` : '';
  const levelText = bookEntry ? levelLabel(bookEntry.targetVisits) : null;
  const jobLine = bookEntry
    ? ` — book job: ${bookEntry.jobStatus} (${bookEntry.totalVisits} visits${levelText ? `, ${levelText}` : ''}${solvedSuffix})`
    : ' — loading book…';

  if (game.isGameOver()) {
    const winner = game.getWinner(); // 0=none, 1=engine BLACK, 2=engine WHITE
    statusEl.textContent = (winner === 0 ? 'Draw' : `${displayColor(winner)} wins!`) + jobLine;
  } else {
    const player = displayColor(game.getCurrentPlayer());
    statusEl.textContent = `${player}'s turn` + jobLine;
  }

  // Offer to re-run the CURRENT position deeper (or, at the same level, a
  // plain re-run) - same idea as the table's per-row action (see
  // renderMovesTable), just for the root instead of one of its candidate
  // moves. Unlike a child move's `expanded`, the root's own jobStatus is
  // never masked by a lingering result, so IN_PROGRESS alone is enough to
  // suppress this - no childInProgress-style helper needed here.
  if (bookEntry && bookEntry.solvedStatus === 'UNSOLVED' && bookEntry.jobStatus !== 'IN_PROGRESS') {
    if (deepeningMoves.has(rootDeepenKey())) {
      statusEl.appendChild(document.createTextNode(' (requeued…)'));
    } else {
      const label = rerunLabel(bookEntry.targetVisits);
      if (label) {
        const btn = document.createElement('button');
        btn.className = 'queue-btn';
        btn.textContent = label;
        btn.addEventListener('click', () => onDeepenRootClick(btn, label));
        statusEl.appendChild(document.createTextNode(' '));
        statusEl.appendChild(btn);
      }
    }
  }
}

// --- Move / history handlers ---
// Drills into `move` as an actual play - shared by clicking an allowed cell
// on the board (see onCellClick) and clicking a move's own link in the
// table (see renderMovesTable). Every table row is already an allowed move
// (see allowedTopMoves()), so this should always succeed there; the legality
// check is still real, not just defensive, for the board's own click path.
function playMove(move) {
  if (game.isGameOver()) return;
  const { x, y } = parseMoveStr(move);
  if (!game.makeMove(x, y)) return; // illegal move (turn/capture/placement rules enforced by the engine)
  moveHistory.push(move);
  redoStack = [];
  loadBookEntry();
}

// Clicking an empty cell already in allowedMoves ("the list" - see
// allowedTopMoves()) plays it (see playMove). Clicking any other empty
// cell - a fresh move the book has never seen, or one that was removed
// (and so has no ghost at all - see renderBoard) - doesn't move the game
// along; it adds that move to the position's allowed-move list instead
// (PUT /pente/book/allowed-moves), which is the only way a move reaches
// the table now.
function onCellClick(x, y) {
  if (game.isGameOver()) return;
  if (game.getStoneAt(x, y) !== 0) return;

  const move = moveStr(x, y);
  if (allowedMoveList().includes(move)) {
    playMove(move);
  } else {
    addAllowedMove(move);
  }
}

// Clears any queuedMoves/deepeningMoves flag left over for this exact
// (position, move) pair - see addAllowedMove/removeAllowedMove. Without
// this, a move that was queued and then removed (or whose search failed)
// before ever actually expanding leaves a stale flag behind forever -
// loadBookEntry's own pruning only ever clears one once the move shows up
// expanded/in-progress, which a removed or failed move never will. Adding
// (or re-adding) that same move later would then show it as already
// "Queued"/"requeued…" even though nothing was actually queued this time.
function clearQueueFlags(move) {
  const key = queueKey(move);
  const hadQueued = queuedMoves.delete(key);
  const hadDeepening = deepeningMoves.delete(key);
  if (hadQueued) saveQueuedMoves();
  if (hadDeepening) saveDeepeningMoves();
}

async function addAllowedMove(move) {
  if (!bookEntry || allowedMovesBusy) return;
  clearQueueFlags(move);
  const allowed = allowedMoveList();
  if (!allowed.includes(move)) allowed.push(move);
  await applyAllowedMoves(allowed, `Adding ${move} to allowed moves…`, `Error adding ${move} to allowed moves`);
}

async function removeAllowedMove(move, btn) {
  if (!bookEntry || allowedMovesBusy) return;
  // Not just a visibility toggle - api/tasks/book.py's set_allowed_moves
  // deletes the move's own subtree outright (its result, its own allowed
  // moves, everything under it) if this position was its only parent -
  // real, possibly expensive computation, not recoverable short of
  // re-running the search. It's only ever *detached* (not deleted) if some
  // other position also has it in its own allowed moves.
  if (!confirm(`Delete ${move} and everything under it (unless it's also reachable from elsewhere)?`)) return;
  btn.disabled = true;
  clearQueueFlags(move);
  const allowed = allowedMoveList().filter(m => m !== move);
  await applyAllowedMoves(allowed, `Removing ${move} from allowed moves…`, `Error removing ${move} from allowed moves`, btn);
}

// `btn` (removeAllowedMove only) gets re-enabled on failure - on success the
// row it belongs to no longer exists after loadBookEntry() re-renders the table.
async function applyAllowedMoves(allowed, pendingMessage, errorPrefix, btn) {
  allowedMovesBusy = true;
  statusEl.textContent = pendingMessage;
  try {
    await putAllowedMoves(moveHistory, allowed);
    await loadBookEntry();
  } catch (e) {
    statusEl.textContent = `${errorPrefix}: ${e.message}`;
    if (btn) btn.disabled = false;
  } finally {
    allowedMovesBusy = false;
  }
}

function undoMove() {
  if (moveHistory.length <= 1) return; // can't undo the forced center opening
  redoStack.push(moveHistory.pop());
  replayHistory();
  loadBookEntry();
}

function redoMove() {
  if (redoStack.length === 0) return;
  const m = redoStack.pop();
  const { x, y } = parseMoveStr(m);
  game.makeMove(x, y);
  moveHistory.push(m);
  loadBookEntry();
}

// Fetches the book entry for the current position, then polls it while a
// search job is running so the table/board pick up progress and eventual
// results without a manual refresh.
async function loadBookEntry() {
  stopPolling();
  syncUrl(); // every moveHistory change (play/undo/redo/reset) funnels through here
  bookEntry = null;
  render();

  const token = ++requestToken;
  try {
    const entry = await fetchBookEntry(moveHistory);
    if (token !== requestToken) return; // superseded by a newer request
    bookEntry = entry;

    // Once a move's own entry confirms it's actually expanded, the locally-queued
    // flag has served its purpose (bridging POST -> the worker picking it up) -
    // drop it so localStorage doesn't grow forever.
    let pruned = false;
    for (const m of allowedTopMoves()) {
      const key = queueKey(m.move);
      if (queuedMoves.has(key) && moveState(m) === 'expanded') {
        queuedMoves.delete(key);
        pruned = true;
      }
    }
    if (pruned) saveQueuedMoves();

    // Same idea for deepeningMoves: once the worker's actually picked up
    // the request, the local flag has served its purpose (bridging POST ->
    // pickup) and childInProgress/jobStatus - real, unmasked signals - take
    // over from here. Unlike queuedMoves above, this can't wait for
    // "expanded" (a re-run's target position was already expanded before
    // the click), and for a same-level re-run childTargetVisits never even
    // changes, so childInProgress is the only reliable signal.
    let deepenPruned = false;
    for (const m of allowedTopMoves()) {
      const key = queueKey(m.move);
      if (deepeningMoves.has(key) && m.childInProgress) {
        deepeningMoves.delete(key);
        deepenPruned = true;
      }
    }
    // The root's own jobStatus is never masked this way (see renderStatus),
    // so it's the same signal used to suppress the button in the first place.
    if (deepeningMoves.has(rootDeepenKey()) && bookEntry.jobStatus === 'IN_PROGRESS') {
      deepeningMoves.delete(rootDeepenKey());
      deepenPruned = true;
    }
    if (deepenPruned) saveDeepeningMoves();
  } catch (e) {
    if (token !== requestToken) return;
    statusEl.textContent = `Error loading book entry: ${e.message}`;
    return;
  }
  render();

  const childPending = allowedTopMoves().some(
    m =>
      moveState(m) === 'queued' ||
      moveState(m) === 'in-progress' ||
      deepeningMoves.has(queueKey(m.move)) ||
      m.childInProgress
  );
  const rootDeepening = deepeningMoves.has(rootDeepenKey());
  if (bookEntry.jobStatus === 'QUEUED' || bookEntry.jobStatus === 'IN_PROGRESS' || childPending || rootDeepening) {
    startPolling();
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setTimeout(loadBookEntry, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

async function onQueueClick(move, btn) {
  btn.disabled = true;
  btn.textContent = 'Queuing…';
  try {
    await queueEvaluation([...moveHistory, move]);
    queuedMoves.add(queueKey(move));
    saveQueuedMoves();
    render();
    startPolling(); // pick up the job's real jobStatus once the worker gets to it
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Queue';
    statusEl.textContent = `Error queuing move: ${e.message}`;
  }
}

// Shared by onDeepenClick (a table row) and onDeepenRootClick (the current
// position) - re-running an already-evaluated position at the currently
// selected level, deeper or the same (see rerunLabel). `key` is
// deepeningMoves' bookkeeping key - queueKey(move) for a row, rootDeepenKey()
// for the root. `label` ('Deepen' or 'Re-run') is just what the button
// should say, both while waiting and if it needs to be restored on failure.
async function requeueEvaluation(moves, key, btn, label) {
  btn.disabled = true;
  btn.textContent = label === 'Re-run' ? 'Re-running…' : 'Requeuing…';
  try {
    await queueEvaluation(moves);
    deepeningMoves.add(key);
    saveDeepeningMoves();
    render();
    startPolling();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = label;
    statusEl.textContent = `Error requeuing: ${e.message}`;
  }
}

function onDeepenClick(move, btn, label) {
  return requeueEvaluation([...moveHistory, move], queueKey(move), btn, label);
}

function onDeepenRootClick(btn, label) {
  return requeueEvaluation(moveHistory, rootDeepenKey(), btn, label);
}

// Reset always starts fresh at the center opening, ignoring the URL (unlike
// the initial load below) - and () => newGame(), not newGame directly, since
// addEventListener would otherwise pass the click event itself as
// initialMoves.
resetBtn.addEventListener('click', () => newGame());
undoBtn.addEventListener('click', undoMove);
redoBtn.addEventListener('click', redoMove);
refreshBtn.addEventListener('click', loadBookEntry);

PenteModule().then(mod => {
  Module = mod;
  newGame(loadMovesFromUrl());
});

// Independent of the game/board above (see loadQueue's own comment) - starts
// right away rather than waiting on PenteModule, and keeps polling for the
// life of the page.
loadQueue();
setInterval(loadQueue, POLL_INTERVAL_MS);
