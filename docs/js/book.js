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
const POLL_INTERVAL_MS = 5000;

const QUEUED_STORAGE_KEY = 'pente-book-queued-moves';

let Module, game;
let moveHistory = [];  // move strings in play order, e.g. ["K10", "L9"]; always starts with the forced center opening
let redoStack = [];    // move strings popped off by undo, replayed by redo
let bookEntry = null;  // last GET /pente/book response for the current position, or null while loading
let queuedMoves = loadQueuedMoves(); // "movesJSON|move" keys the user has queued, persisted across reloads - see queueKey()
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
  const res = await fetch(`${API_BASE}/pente/book?${params}`);
  if (!res.ok) throw new Error(`GET /pente/book failed: ${res.status}`);
  return res.json();
}

async function queueEvaluation(moves) {
  const res = await fetch(`${API_BASE}/pente/book`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves }),
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

// Key identifying "this candidate move, from this position" - used to
// remember which moves the user has queued (see renderMovesTable).
function queueKey(move) {
  return JSON.stringify(moveHistory) + '|' + move;
}

// Queued moves persist in localStorage (not just in memory) so a page reload
// doesn't forget a move was queued and offer to queue it again - useful since
// a real search can take a long time to actually start (see FULL_SEARCH_ITERATIONS
// in api/tasks/book.py) or finish. Falls back to an empty set if storage is
// unavailable (e.g. private browsing).
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
      cell.classList.remove('last-move');
      cell.style.boxShadow = '';
      const stoneVal = game.getStoneAt(x, y); // 0=empty, 1=black, 2=white
      cell.classList.toggle('occupied', stoneVal !== 0);
      if (stoneVal === 1 || stoneVal === 2) {
        const stone = document.createElement('div');
        stone.className = 'stone ' + (stoneVal === 1 ? 'black' : 'white');
        cell.appendChild(stone);
      }
    }
  }

  const last = moveHistory[moveHistory.length - 1];
  if (last) {
    const { x, y } = parseMoveStr(last);
    cells[y * BOARD_SIZE + x].classList.add('last-move');
  }

  // Ghost overlay: every candidate reply the book knows about for the side
  // to move, numbered by rank - allowed or not (see visibleMoves()). A
  // not-currently-allowed one is dimmer still, distinguishing it, but no
  // longer disappears outright the way removeAllowedMove used to make it.
  // Not-yet-queued moves are faint; queued/in-progress/expanded ones are
  // shown solid but still lighter than a real stone.
  if (bookEntry) {
    const toMoveColor = game.getCurrentPlayer() === 1 ? 'black' : 'white';
    const moves = visibleMoves();
    // Priors come straight from the engine's own heuristic policy (see
    // HeuristicEvaluator::evaluatePolicy) - a move it flagged as tactically
    // relevant (open three, capture threat/defense, four threat, etc.) gets
    // a nonzero prior, a "quiet" move gets exactly 0. Darkening scales with
    // prior relative to the strongest one on the board, so the sharpest
    // threats stand out most - not touching the heuristic itself, just
    // reflecting what it already computed.
    const maxPrior = Math.max(0, ...moves.map(m => m.prior));
    moves.forEach((m, i) => {
      const { x, y } = parseMoveStr(m.move);
      const cell = cells[y * BOARD_SIZE + x];
      if (cell.classList.contains('occupied')) return; // shouldn't happen, but never draw a ghost over a real stone
      if (m.prior > 0 && maxPrior > 0) {
        cell.style.boxShadow = `inset 0 0 0 999px rgba(0,0,0,${0.35 * m.prior / maxPrior})`;
      }
      const expanded = moveState(m) !== 'none';
      const ghost = document.createElement('div');
      ghost.className = 'stone ghost ' + toMoveColor + (expanded ? ' expanded' : '') + (m.isAllowed ? '' : ' not-allowed');
      ghost.textContent = String(i + 1);
      cell.appendChild(ghost);
    });
  }
}

// Every candidate move known for this position, allowed or not. isAllowed
// is a proof-relevance flag (see api/kv_store.py's compute_book_solved_status),
// not a visibility filter - it used to double as one here, which meant a
// fresh evaluation's own best move could come back marked isAllowed:false
// (nobody had added it to allowedMoves yet, since nobody could have known
// about it beforehand) and silently vanish from view. Not-allowed moves are
// still shown (see renderBoard/renderMovesTable), just visually distinguished.
function visibleMoves() {
  return bookEntry ? bookEntry.topMoves : [];
}

// The subset actually in the position's allowedMoves right now - what the
// moves table lists (see renderMovesTable) and the base for building a new
// list to PUT (see addAllowedMove/removeAllowedMove). Deliberately separate
// from visibleMoves(): building the PUT payload from the *unfiltered* list
// would silently allow everything just by adding one move.
function allowedTopMoves() {
  return bookEntry ? bookEntry.topMoves.filter(m => m.isAllowed) : [];
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
    // avgValue is null for a manually-added move the engine hasn't ranked
    // (or hasn't run on) yet - see _to_book_entry in api/routers/book.py.
    const avgValue = m.avgValue === null ? 'N/A' : m.avgValue.toFixed(3);
    row.innerHTML = `<td>${i + 1}</td><td><a href="#" class="move-link">${m.move}</a></td><td>${avgValue}</td><td>${m.prior.toFixed(3)}</td><td>${m.childMoveCount}</td><td></td><td></td>`;
    const [statusCell, allowedCell] = [row.children[5], row.children[6]];

    // Same as clicking this move's cell on the board (see onCellClick) -
    // every row here is already allowed, so this always plays it.
    row.querySelector('.move-link').addEventListener('click', e => {
      e.preventDefault();
      playMove(m.move);
    });

    // One merged column: the search's own result once it's settled
    // something (a real answer beats a progress indicator), otherwise
    // whatever's actionable right now - the same states the old separate
    // Status/Action columns showed, just never both at once.
    if (m.status !== 'UNSOLVED') {
      statusCell.textContent = m.status;
    } else {
      const state = moveState(m);
      if (state === 'expanded') {
        statusCell.textContent = 'Expanded';
      } else if (state === 'in-progress') {
        statusCell.textContent = 'In progress…';
      } else if (state === 'queued') {
        statusCell.textContent = 'Queued';
      } else {
        const btn = document.createElement('button');
        btn.className = 'queue-btn';
        btn.textContent = 'Queue';
        btn.addEventListener('click', () => onQueueClick(m.move, btn));
        statusCell.appendChild(btn);
      }
    }

    // Always a remove control - every row here is already allowed (see
    // allowedTopMoves()), so there's never an add case to offer.
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.title = `Delete ${m.move} (and its subtree, unless reachable elsewhere)`;
    removeBtn.textContent = '🗑';
    removeBtn.addEventListener('click', () => removeAllowedMove(m.move, removeBtn));
    allowedCell.appendChild(removeBtn);

    movesBody.appendChild(row);
  });
}

function renderStatus() {
  capturesEl.textContent = `Captures — Black: ${game.getBlackCaptures()}, White: ${game.getWhiteCaptures()}`;

  // Only worth reporting the book-level solved status once it's actually
  // settled - UNSOLVED is the default/common case, not news.
  const solvedSuffix = bookEntry && bookEntry.solvedStatus !== 'UNSOLVED' ? `, ${bookEntry.solvedStatus}` : '';
  const jobLine = bookEntry
    ? ` — book job: ${bookEntry.jobStatus} (${bookEntry.totalVisits} visits${solvedSuffix})`
    : ' — loading book…';

  if (game.isGameOver()) {
    const winner = game.getWinner(); // 0=none, 1=black, 2=white
    statusEl.textContent = (winner === 1 ? 'Black wins!' : winner === 2 ? 'White wins!' : 'Draw') + jobLine;
    return;
  }
  const player = game.getCurrentPlayer() === 1 ? 'Black' : 'White';
  statusEl.textContent = `${player}'s turn` + jobLine;
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
// allowedTopMoves()) plays it (see playMove). Clicking any other empty cell -
// a fresh move the book has never seen, or one the engine already suggests
// but nobody's approved yet (shown as a dim ghost - see renderBoard) -
// doesn't move the game along; it adds that move to the position's
// allowed-move list instead (PUT /pente/book/allowed-moves), which is the
// only way a move reaches the table now.
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

async function addAllowedMove(move) {
  if (!bookEntry || allowedMovesBusy) return;
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
    for (const m of visibleMoves()) {
      const key = queueKey(m.move);
      if (queuedMoves.has(key) && moveState(m) === 'expanded') {
        queuedMoves.delete(key);
        pruned = true;
      }
    }
    if (pruned) saveQueuedMoves();
  } catch (e) {
    if (token !== requestToken) return;
    statusEl.textContent = `Error loading book entry: ${e.message}`;
    return;
  }
  render();

  const childPending = visibleMoves().some(m => moveState(m) === 'queued' || moveState(m) === 'in-progress');
  if (bookEntry.jobStatus === 'QUEUED' || bookEntry.jobStatus === 'IN_PROGRESS' || childPending) {
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
