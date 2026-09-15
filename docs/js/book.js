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
let childStatuses = new Map(); // move -> BookEntry for that child position (current moveHistory + move) - see fetchChildStatuses
let queuedMoves = loadQueuedMoves(); // "movesJSON|move" keys the user has queued, persisted across reloads - see queueKey()
let requestToken = 0;  // guards against a stale fetch response overwriting a newer one
let pollTimer = null;
let addingAllowedMove = false; // guards against overlapping PUT /allowed-moves calls - see addAllowedMove

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

// A candidate move's real state comes from its own book_db entry, not the
// (currently always "false" - see book.py's _to_book_entry TODO) `expanded`
// field on the parent's topMoves. Fetching it directly is safe here: every
// move in bookEntry.topMoves was already recorded as its own entry by
// add_parent_edge when the parent's search completed, so this never risks
// triggering a fresh (potentially 900M-iteration) search the way GETting an
// arbitrary unseen position would.
async function fetchChildStatuses(topMoves) {
  const entries = await Promise.all(
    topMoves.map(m =>
      fetchBookEntry([...moveHistory, m.move])
        .then(entry => [m.move, entry])
        .catch(() => [m.move, null])
    )
  );
  return new Map(entries.filter(([, entry]) => entry !== null));
}

// Where a candidate move actually stands, combining its own book_db entry
// with the locally-queued flag (which bridges the gap between a successful
// POST and the worker actually picking the job up - see onQueueClick).
function moveState(move) {
  const child = childStatuses.get(move);
  if (child && (child.totalVisits > 0 || child.bestMove !== null)) return 'expanded';
  if (child && child.jobStatus === 'IN_PROGRESS') return 'in-progress';
  if ((child && child.jobStatus === 'QUEUED') || queuedMoves.has(queueKey(move))) return 'queued';
  return 'none';
}

// --- Game setup ---
function newGame() {
  if (game) game.delete();
  game = new Module.Game(BOARD_SIZE, 1); // simulations param is unused - this Game is only used here for rule enforcement
  moveHistory = [];
  redoStack = [];
  game.makeMove(CENTER, CENTER);
  moveHistory.push(moveStr(CENTER, CENTER));
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

  // Ghost overlay: the book's candidate replies for the side to move, numbered
  // by rank. Not-yet-queued moves are faint; queued/in-progress/expanded ones
  // are shown solid but still lighter than a real stone.
  if (bookEntry) {
    const toMoveColor = game.getCurrentPlayer() === 1 ? 'black' : 'white';
    bookEntry.topMoves.forEach((m, i) => {
      const { x, y } = parseMoveStr(m.move);
      const cell = cells[y * BOARD_SIZE + x];
      if (cell.classList.contains('occupied')) return; // shouldn't happen, but never draw a ghost over a real stone
      const expanded = moveState(m.move) !== 'none';
      const ghost = document.createElement('div');
      ghost.className = 'stone ghost ' + toMoveColor + (expanded ? ' expanded' : '');
      ghost.textContent = String(i + 1);
      cell.appendChild(ghost);
    });
  }
}

function renderMovesTable() {
  movesBody.innerHTML = '';
  if (!bookEntry) return;
  bookEntry.topMoves.forEach((m, i) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${i + 1}</td><td>${m.move}</td><td>${m.avgValue.toFixed(3)}</td><td>${m.status}</td><td></td>`;
    const actionCell = row.lastElementChild;

    const state = moveState(m.move);
    if (state === 'expanded') {
      actionCell.textContent = 'Expanded';
    } else if (state === 'in-progress') {
      actionCell.textContent = 'In progress…';
    } else if (state === 'queued') {
      actionCell.textContent = 'Queued';
    } else {
      const btn = document.createElement('button');
      btn.className = 'queue-btn';
      btn.textContent = 'Queue';
      btn.addEventListener('click', () => onQueueClick(m.move, btn));
      actionCell.appendChild(btn);
    }
    movesBody.appendChild(row);
  });
}

function renderStatus() {
  capturesEl.textContent = `Captures — Black: ${game.getBlackCaptures()}, White: ${game.getWhiteCaptures()}`;

  const jobLine = bookEntry
    ? ` — book job: ${bookEntry.jobStatus} (${bookEntry.totalVisits} visits, ${bookEntry.solvedStatus})`
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
// Clicking an empty cell that's already a book candidate for this position
// plays it as an actual move (drilling into that line). Clicking an empty
// cell that isn't one of the book's candidates doesn't move the game along
// at all - it just proposes that move for consideration by adding it to the
// position's allowed-move list (PUT /pente/book/allowed-moves).
function onCellClick(x, y) {
  if (game.isGameOver()) return;
  if (game.getStoneAt(x, y) !== 0) return;

  const move = moveStr(x, y);
  const isCandidate = bookEntry && bookEntry.topMoves.some(m => m.move === move);
  if (isCandidate) {
    if (!game.makeMove(x, y)) return; // illegal move (turn/capture/placement rules enforced by the engine)
    moveHistory.push(move);
    redoStack = [];
    loadBookEntry();
  } else {
    addAllowedMove(move);
  }
}

async function addAllowedMove(move) {
  if (!bookEntry || addingAllowedMove) return;
  addingAllowedMove = true;
  const allowed = bookEntry.topMoves.filter(m => m.isAllowed).map(m => m.move);
  if (!allowed.includes(move)) allowed.push(move);

  statusEl.textContent = `Adding ${move} to allowed moves…`;
  try {
    await putAllowedMoves(moveHistory, allowed);
    await loadBookEntry();
  } catch (e) {
    statusEl.textContent = `Error adding ${move} to allowed moves: ${e.message}`;
  } finally {
    addingAllowedMove = false;
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
  bookEntry = null;
  childStatuses = new Map();
  render();

  const token = ++requestToken;
  try {
    const entry = await fetchBookEntry(moveHistory);
    if (token !== requestToken) return; // superseded by a newer request
    bookEntry = entry;
    childStatuses = await fetchChildStatuses(entry.topMoves);
    if (token !== requestToken) return;

    // Once a move's own entry confirms it's actually expanded, the locally-queued
    // flag has served its purpose (bridging POST -> the worker picking it up) -
    // drop it so localStorage doesn't grow forever.
    let pruned = false;
    for (const m of entry.topMoves) {
      const key = queueKey(m.move);
      if (queuedMoves.has(key) && moveState(m.move) === 'expanded') {
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

  const childPending = bookEntry.topMoves.some(m => moveState(m.move) === 'queued' || moveState(m.move) === 'in-progress');
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

resetBtn.addEventListener('click', newGame);
undoBtn.addEventListener('click', undoMove);
redoBtn.addEventListener('click', redoMove);
refreshBtn.addEventListener('click', loadBookEntry);

PenteModule().then(mod => {
  Module = mod;
  newGame();
});
