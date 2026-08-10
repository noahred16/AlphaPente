const EFFORT_SIMULATIONS = { low: 3000, medium: 10000, high: 30000, ultra: 100000 };
const BOOK_URL = { 4: 'data/book4x4.bin.gz' }; // board size -> gzipped solved-book asset

let Module, game, boardSize;
let lastTopMoves = null; // kept visible (table + highlight) until the next AI search
let lastAiMove = null; // {x, y} of the AI's most recent move, kept highlighted until its next move
let moveHistory = []; // [{x, y}, ...] in play order, Black first; used to replay after undo
let aiPending = false; // true from when the AI's turn is scheduled until its move commits
const bookBytesCache = {}; // board size -> fetched Uint8Array, so switching sizes doesn't re-fetch

const loadingEl = document.getElementById('loading');
const appEl = document.getElementById('app');
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const capturesEl = document.getElementById('captures');
const topMovesHead = document.querySelector('#top-moves thead');
const topMovesBody = document.querySelector('#top-moves tbody');
const resetBtn = document.getElementById('reset');
const undoBtn = document.getElementById('undo');
const settingsBtn = document.getElementById('settings-btn');
const settingsDialog = document.getElementById('settings-dialog');
const settingsCloseBtn = document.getElementById('settings-close');
const boardSizeSelect = document.getElementById('board-size');
const pageTitleEl = document.getElementById('page-title');
const pageHeadingEl = document.getElementById('page-heading');

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function getBoardSizeSetting() {
  return Number(boardSizeSelect.value);
}

function getEffortSimulations() {
  const level = document.querySelector('input[name="effort"]:checked').value;
  return EFFORT_SIMULATIONS[level];
}

// Fill in each effort label with its iteration count, e.g. "Low (3K)", read
// straight from EFFORT_SIMULATIONS so labels can't drift out of sync with it.
function labelEffortButtons() {
  document.querySelectorAll('input[name="effort"]').forEach(input => {
    const n = EFFORT_SIMULATIONS[input.value];
    const count = n >= 1000 ? `${n / 1000}K` : n;
    const span = input.nextElementSibling;
    span.textContent = `${span.textContent} (${count})`;
  });
}

// Fetches (or returns the cached copy of) the solved book for `size`, with a
// live percentage in #loading - a plain "Loading..." message would look just
// as stuck as no message at all for a multi-MB download on a slow
// connection. Aborts (and surfaces an error - see newGame()) after 60s of no
// response at all, so a genuinely stalled connection doesn't hang forever
// indistinguishably from "still working."
//
// The book ships gzipped (see docs/data/book4x4.bin.gz - roughly half the
// size of the raw .bin on top of the on-disk format itself already being
// tight, see PositionBook.cpp) and is decompressed client-side via
// DecompressionStream, widely supported since 2023 (Chrome/Edge 80+,
// Firefox 113+, Safari 16.4+). No older-browser fallback: on an unsupported
// browser `new DecompressionStream(...)` throws, which newGame()'s try/catch
// already turns into a visible error message rather than a silent hang -
// still a strictly better failure mode than the bug this is fixing.
async function fetchBookBytes(size) {
  if (bookBytesCache[size]) return bookBytesCache[size];

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  let resp;
  try {
    resp = await fetch(BOOK_URL[size], { signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
  if (!resp.ok) {
    throw new Error(`Fetching ${BOOK_URL[size]} failed: HTTP ${resp.status}`);
  }

  const total = Number(resp.headers.get('Content-Length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      const pct = Math.round((received / total) * 100);
      setLoading(`Loading solved ${size} x ${size} book… ${pct}%`);
    } else {
      setLoading(`Loading solved ${size} x ${size} book… ${(received / 1e6).toFixed(1)}MB`);
    }
  }
  setLoading(`Unpacking solved ${size} x ${size} book…`);
  const compressed = new Blob(chunks);
  const decompressedStream = compressed.stream().pipeThrough(new DecompressionStream('gzip'));
  const bytes = new Uint8Array(await new Response(decompressedStream).arrayBuffer());

  bookBytesCache[size] = bytes;
  return bytes;
}

function setLoading(text) {
  loadingEl.textContent = text;
  loadingEl.classList.remove('hidden');
  appEl.classList.add('hidden');
}

function clearLoading() {
  loadingEl.classList.add('hidden');
  appEl.classList.remove('hidden');
}

// Any failure below (network error, HTTP error status, a stalled connection
// past fetchBookBytes()'s 60s abort, or a WASM-side exception) used to leave
// the loading message showing forever with no indication anything went
// wrong - the actual bug behind "gets stuck on the loading message". Now
// surfaced as a visible, specific message instead of a silent hang.
async function newGame() {
  const size = getBoardSizeSetting();
  setLoading(`Loading ${size} x ${size}…`);

  try {
    if (game) game.delete();
    game = new Module.Game(size, getEffortSimulations());
    boardSize = game.getBoardSize();

    if (BOOK_URL[boardSize]) {
      const bytes = await fetchBookBytes(boardSize);
      game.loadBookFromBytes(bytes);
    }

    const usingBook = game.usingBook();
    const title = `${boardSize} x ${boardSize} Pente` + (usingBook ? ' (solved)' : '');
    pageTitleEl.textContent = title;
    pageHeadingEl.textContent = title;
    lastTopMoves = null;
    lastAiMove = null;
    moveHistory = [];
    aiPending = false;
    buildBoard();
    buildTopMovesHeader(usingBook);
    render();
    clearLoading();
  } catch (err) {
    console.error('newGame failed:', err);
    setLoading(`Failed to load: ${err.message || err}. Reload the page to retry.`);
  }
}

// Book-backed boards show every legal reply's exact outcome + moves-to-result
// instead of MCTS's visit/value/PUCT stats, which don't apply to an exact
// lookup - see WasmGame::getTopMoves() (wasm/PenteWasm.cpp) for the C++ side.
function buildTopMovesHeader(usingBook) {
  topMovesHead.innerHTML = usingBook
    ? '<tr><th>Move</th><th>Result</th><th>Moves to result</th></tr>'
    : '<tr><th>Move</th><th>Visits</th><th>Avg Value</th><th>PUCT</th><th>Status</th></tr>';
}

function buildBoard() {
  boardEl.innerHTML = '';
  boardEl.style.setProperty('--board-size', boardSize);
  for (let y = 0; y < boardSize; y++) {
    for (let x = 0; x < boardSize; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.addEventListener('click', () => onCellClick(x, y));
      boardEl.appendChild(cell);
    }
  }
}

// topMoves: optional array of {x, y, visits, value} to highlight (AI search overlay)
function render(topMoves) {
  const cells = boardEl.children;
  for (let y = 0; y < boardSize; y++) {
    for (let x = 0; x < boardSize; x++) {
      const cell = cells[y * boardSize + x];
      cell.innerHTML = '';
      cell.classList.remove('top-move');
      cell.classList.remove('last-ai-move');
      const stoneVal = game.getStoneAt(x, y); // 0=empty, 1=black, 2=white
      cell.classList.toggle('occupied', stoneVal !== 0);
      if (stoneVal === 1 || stoneVal === 2) {
        const stone = document.createElement('div');
        stone.className = 'stone ' + (stoneVal === 1 ? 'black' : 'white');
        cell.appendChild(stone);
      }
    }
  }
  if (topMoves) {
    for (const m of topMoves) {
      cells[m.y * boardSize + m.x].classList.add('top-move');
    }
  }
  if (lastAiMove) {
    cells[lastAiMove.y * boardSize + lastAiMove.x].classList.add('last-ai-move');
  }
  renderTopMoves(topMoves);
  updateStatus();
}

// topMoves: same array passed to render(), or undefined/empty to clear the table.
function renderTopMoves(topMoves) {
  topMovesBody.innerHTML = '';
  if (!topMoves) return;
  const usingBook = game.usingBook();
  for (const m of topMoves) {
    const row = document.createElement('tr');
    row.innerHTML = usingBook
      ? `<td>(${m.x}, ${m.y})</td><td>${m.status}</td><td>${m.depth}</td>`
      : `<td>(${m.x}, ${m.y})</td><td>${m.visits}</td>` +
        `<td>${m.value.toFixed(3)}</td><td>${m.puct.toFixed(3)}</td><td>${m.status}</td>`;
    topMovesBody.appendChild(row);
  }
}

function updateStatus() {
  capturesEl.textContent = `Captures — Black: ${game.getBlackCaptures()}, White: ${game.getWhiteCaptures()}`;
  if (game.isGameOver()) {
    const winner = game.getWinner(); // 0=none, 1=black, 2=white
    statusEl.textContent = winner === 1 ? 'Black wins!' : winner === 2 ? 'White wins!' : 'Draw';
    return;
  }
  const player = game.getCurrentPlayer() === 1 ? 'Black' : 'White';
  statusEl.textContent = `${player}'s turn`;
}

function onCellClick(x, y) {
  if (game.isGameOver()) return;
  if (getMode() === 'ai' && game.getCurrentPlayer() !== 1) return; // AI (White) is moving
  if (game.getStoneAt(x, y) !== 0) return; // cell already has a stone
  if (!game.makeMove(x, y)) return;
  moveHistory.push({ x, y });
  lastTopMoves = null;
  render();

  if (getMode() === 'ai' && !game.isGameOver()) {
    statusEl.textContent = 'AI is thinking...';
    aiPending = true;
    setTimeout(aiTurn, 30);
  }
}

// Search first (without applying), show the top candidate moves briefly, then commit.
// The table + highlight are left showing (via lastTopMoves) after the move commits,
// until the next AI search replaces them or a human move clears them.
function aiTurn() {
  const move = game.computeAIMove();
  if (move.x < 0) { aiPending = false; lastTopMoves = null; render(); return; } // no moves left (draw)
  // A solved book covers every legal reply, not just the top few - show all
  // of them (25 comfortably covers any board size up to 5x5).
  lastTopMoves = game.getTopMoves(game.usingBook() ? 25 : 10);
  render(lastTopMoves);
  setTimeout(() => {
    game.makeMove(move.x, move.y);
    moveHistory.push({ x: move.x, y: move.y });
    lastAiMove = { x: move.x, y: move.y };
    aiPending = false;
    render(lastTopMoves);
  }, 500);
}

// Undoes the last full turn: 1 move in PvP, or the AI's move plus the human's
// move that provoked it in Player vs AI. Disabled while the AI is mid-turn
// (search running or its move about to commit) to avoid racing that timeout.
function undoMove() {
  if (aiPending) return; // AI move in flight; wait for it to commit
  const removeCount = getMode() === 'ai' ? 2 : 1;
  if (moveHistory.length < removeCount) return;
  moveHistory.length -= removeCount;

  game.reset();
  lastTopMoves = null;
  lastAiMove = null;
  for (const m of moveHistory) game.makeMove(m.x, m.y);
  if (getMode() === 'ai' && moveHistory.length % 2 === 0 && moveHistory.length > 0) {
    lastAiMove = moveHistory[moveHistory.length - 1]; // last move replayed was White's (AI's)
  }
  render();
}

labelEffortButtons();
resetBtn.addEventListener('click', newGame);
undoBtn.addEventListener('click', undoMove);
settingsBtn.addEventListener('click', () => settingsDialog.classList.add('open'));
settingsCloseBtn.addEventListener('click', () => settingsDialog.classList.remove('open'));
boardSizeSelect.addEventListener('change', newGame);
document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', newGame));
document.querySelectorAll('input[name="effort"]').forEach(r =>
  r.addEventListener('change', () => game.setSimulations(getEffortSimulations())));

// Same class of bug as newGame()'s try/catch (see its comment): this outer
// promise/call had no error handling at all, so a failure here - PenteModule
// undefined (wasm/pente.js itself failed to load/parse), a WASM
// instantiation error, anything - left "Loading engine…" showing forever
// with nothing in the UI to explain why. try/catch covers a synchronous
// throw from the PenteModule() call itself (e.g. it not being a function at
// all); .catch() covers the promise it returns actually rejecting.
setLoading('Loading engine…');
function failEngineLoad(err) {
  console.error('Failed to load the WASM engine:', err);
  setLoading(`Failed to load engine: ${err && err.message ? err.message : err}. Reload the page to retry.`);
}
try {
  PenteModule().then(mod => {
    Module = mod;
    newGame();
  }).catch(failEngineLoad);
} catch (err) {
  failEngineLoad(err);
}
