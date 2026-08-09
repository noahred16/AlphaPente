// Bump this to switch board size later (WasmGame takes boardSize as a
// constructor param, so nothing else here needs to change).
const BOARD_SIZE = 5;
const EFFORT_SIMULATIONS = { low: 3000, medium: 10000, high: 30000 };

let Module, game, boardSize;
let lastTopMoves = null; // kept visible (table + highlight) until the next AI search
let lastAiMove = null; // {x, y} of the AI's most recent move, kept highlighted until its next move
let moveHistory = []; // [{x, y}, ...] in play order, Black first; used to replay after undo
let aiPending = false; // true from when the AI's turn is scheduled until its move commits

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const capturesEl = document.getElementById('captures');
const topMovesBody = document.querySelector('#top-moves tbody');
const resetBtn = document.getElementById('reset');
const undoBtn = document.getElementById('undo');
const settingsBtn = document.getElementById('settings-btn');
const settingsDialog = document.getElementById('settings-dialog');
const settingsCloseBtn = document.getElementById('settings-close');

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
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

function newGame() {
  if (game) game.delete();
  game = new Module.Game(BOARD_SIZE, getEffortSimulations());
  boardSize = game.getBoardSize();
  lastTopMoves = null;
  lastAiMove = null;
  moveHistory = [];
  aiPending = false;
  buildBoard();
  render();
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
  for (const m of topMoves) {
    const row = document.createElement('tr');
    row.innerHTML = `<td>(${m.x}, ${m.y})</td><td>${m.visits}</td>` +
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
  lastTopMoves = game.getTopMoves(10);
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
document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', newGame));
document.querySelectorAll('input[name="effort"]').forEach(r =>
  r.addEventListener('change', () => game.setSimulations(getEffortSimulations())));

PenteModule().then(mod => {
  Module = mod;
  newGame();
});
