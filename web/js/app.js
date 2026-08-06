// Bump this to switch board size later (WasmGame takes boardSize as a
// constructor param, so nothing else here needs to change).
const BOARD_SIZE = 5;
const EFFORT_SIMULATIONS = { low: 3000, medium: 10000, high: 30000 };

let Module, game, boardSize;

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const resetBtn = document.getElementById('reset');

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function getEffortSimulations() {
  const level = document.querySelector('input[name="effort"]:checked').value;
  return EFFORT_SIMULATIONS[level];
}

function newGame() {
  if (game) game.delete();
  game = new Module.Game(BOARD_SIZE, getEffortSimulations());
  boardSize = game.getBoardSize();
  buildBoard();
  render();
}

function buildBoard() {
  boardEl.innerHTML = '';
  boardEl.style.gridTemplateColumns = `repeat(${boardSize}, 40px)`;
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
  updateStatus();
}

function updateStatus() {
  if (game.isGameOver()) {
    const winner = game.getWinner(); // 0=none, 1=black, 2=white
    statusEl.textContent = winner === 1 ? 'Black wins!' : winner === 2 ? 'White wins!' : 'Draw';
    return;
  }
  const player = game.getCurrentPlayer() === 1 ? 'Black' : 'White';
  statusEl.textContent =
    `${player}'s turn — Captures: Black ${game.getBlackCaptures()}, White ${game.getWhiteCaptures()}`;
}

function onCellClick(x, y) {
  if (game.isGameOver()) return;
  if (getMode() === 'ai' && game.getCurrentPlayer() !== 1) return; // AI (White) is moving
  if (game.getStoneAt(x, y) !== 0) return; // cell already has a stone
  if (!game.makeMove(x, y)) return;
  render();

  if (getMode() === 'ai' && !game.isGameOver()) {
    statusEl.textContent = 'AI is thinking...';
    setTimeout(aiTurn, 30);
  }
}

// Search first (without applying), show the top candidate moves briefly, then commit.
function aiTurn() {
  const move = game.computeAIMove();
  if (move.x < 0) { render(); return; } // no moves left (draw)
  const topMoves = game.getTopMoves(5);
  render(topMoves);
  setTimeout(() => {
    game.makeMove(move.x, move.y);
    render();
  }, 500);
}

resetBtn.addEventListener('click', newGame);
document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', newGame));
document.querySelectorAll('input[name="effort"]').forEach(r =>
  r.addEventListener('change', () => game.setSimulations(getEffortSimulations())));

PenteModule().then(mod => {
  Module = mod;
  newGame();
});
