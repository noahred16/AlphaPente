const BOARD_SIZE = 19;
const CELL = 28;
const MARGIN = 28;
const SVG_SIZE = MARGIN * 2 + (BOARD_SIZE - 1) * CELL;

const state = {
  page: 1,
  pageSize: 20,
  totalGames: 0,
  game: null,   // currently loaded game detail
  ply: 0,       // currently displayed ply index within state.game.plies
};

// Mirrors GameUtils::displayMove (apps/GameUtils.cpp) — skip 'I' entirely.
function displayMove(x, y) {
  let col = 'A'.charCodeAt(0) + x;
  if (col >= 'I'.charCodeAt(0)) col++;
  return String.fromCharCode(col) + (y + 1);
}

function boardX(x) { return MARGIN + x * CELL; }
function boardY(y) { return MARGIN + (BOARD_SIZE - 1 - y) * CELL; }

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderBoardGrid(svg) {
  svg.setAttribute('viewBox', `0 0 ${SVG_SIZE} ${SVG_SIZE}`);
  for (let i = 0; i < BOARD_SIZE; i++) {
    svg.appendChild(svgEl('line', {
      x1: boardX(0), y1: boardY(i), x2: boardX(BOARD_SIZE - 1), y2: boardY(i),
      stroke: '#0003', 'stroke-width': 1,
    }));
    svg.appendChild(svgEl('line', {
      x1: boardX(i), y1: boardY(0), x2: boardX(i), y2: boardY(BOARD_SIZE - 1),
      stroke: '#0003', 'stroke-width': 1,
    }));
  }
}

function renderPly(game, plyIndex) {
  const svg = document.getElementById('board');
  svg.innerHTML = '';
  renderBoardGrid(svg);

  const ply = game.plies[plyIndex];

  for (const [x, y] of ply.black_stones)
    svg.appendChild(svgEl('circle', { cx: boardX(x), cy: boardY(y), r: CELL * 0.42, fill: 'var(--black-stone)' }));
  for (const [x, y] of ply.white_stones)
    svg.appendChild(svgEl('circle', { cx: boardX(x), cy: boardY(y), r: CELL * 0.42, fill: 'var(--white-stone)' }));

  if (ply.move) {
    svg.appendChild(svgEl('circle', {
      cx: boardX(ply.move.x), cy: boardY(ply.move.y), r: CELL * 0.42,
      fill: 'none', stroke: 'var(--last-move)', 'stroke-width': 3,
    }));
    for (const [x, y] of ply.move.captured) {
      svg.appendChild(svgEl('line', { x1: boardX(x) - 8, y1: boardY(y) - 8, x2: boardX(x) + 8, y2: boardY(y) + 8, stroke: 'var(--captured)', 'stroke-width': 3 }));
      svg.appendChild(svgEl('line', { x1: boardX(x) - 8, y1: boardY(y) + 8, x2: boardX(x) + 8, y2: boardY(y) - 8, stroke: 'var(--captured)', 'stroke-width': 3 }));
    }
  }

  // Ghost markers: what the search considered next from this position.
  for (const cand of ply.top_policy) {
    const r = 4 + cand.p * (CELL * 0.35);
    svg.appendChild(svgEl('circle', {
      cx: boardX(cand.x), cy: boardY(cand.y), r,
      fill: 'var(--policy-ghost)', opacity: 0.55,
    }));
  }

  document.getElementById('ply-label').textContent = `${plyIndex} / ${game.plies.length - 1}`;
  document.getElementById('step-prev').disabled = plyIndex === 0;
  document.getElementById('step-first').disabled = plyIndex === 0;
  document.getElementById('step-next').disabled = plyIndex === game.plies.length - 1;
  document.getElementById('step-last').disabled = plyIndex === game.plies.length - 1;

  highlightMoveRow(plyIndex);
  renderPlyDetail(ply);
}

function renderPlyDetail(ply) {
  const detail = document.getElementById('ply-detail');
  const topPolicy = ply.top_policy.map(c => `${c.display} (${c.p.toFixed(2)})`).join(', ');
  const moveLine = ply.move
    ? `<b>${ply.move.mover}</b> played <b>${ply.move.display}</b>${ply.move.captured.length ? ` (captured ${ply.move.captured.map(([x, y]) => displayMove(x, y)).join(', ')})` : ''}`
    : '<b>game start</b>';
  detail.innerHTML = `
    ${moveLine}<br>
    z=${ply.z.toFixed(2)} rootQ=${ply.root_q.toFixed(2)} entropy=${ply.entropy.toFixed(3)}<br>
    captures: black=${ply.black_captures} white=${ply.white_captures}<br>
    next-to-move top: ${topPolicy}
  `;
}

function highlightMoveRow(plyIndex) {
  for (const row of document.querySelectorAll('#move-table-body tr'))
    row.classList.toggle('selected', Number(row.dataset.ply) === plyIndex);
}

function renderMoveTable(game) {
  const body = document.getElementById('move-table-body');
  body.innerHTML = '';
  game.plies.forEach((ply, i) => {
    const row = document.createElement('tr');
    row.dataset.ply = i;
    const moveStr = ply.move ? `${ply.move.mover[0].toUpperCase()} ${ply.move.display}` : 'start';
    row.innerHTML = `<td>${i}</td><td>${moveStr}</td><td>${ply.move && ply.move.captured.length ? ply.move.captured.length : ''}</td>
                      <td>${ply.z.toFixed(2)}</td><td>${ply.root_q.toFixed(2)}</td><td>${ply.entropy.toFixed(2)}</td>`;
    row.addEventListener('click', () => { state.ply = i; renderPly(state.game, i); });
    body.appendChild(row);
  });
}

async function loadGame(index) {
  state.game = await fetchJSON(`/api/games/${index}`);
  state.ply = state.game.plies.length - 1;  // land on the final position by default
  document.getElementById('game-label').textContent =
    `Game ${state.game.index}${state.game.partial ? ' (partial)' : ''} — ${state.game.plies.length} plies — winner: ${state.game.winner ?? '?'}`;
  renderMoveTable(state.game);
  renderPly(state.game, state.ply);

  for (const row of document.querySelectorAll('#game-list-body tr'))
    row.classList.toggle('selected', Number(row.dataset.index) === index);
}

async function loadGameList() {
  const data = await fetchJSON(`/api/games?page=${state.page}&page_size=${state.pageSize}`);
  state.totalGames = data.total_games;
  document.getElementById('run-name').textContent = `run: ${data.run}`;

  const body = document.getElementById('game-list-body');
  body.innerHTML = '';
  for (const g of data.games) {
    const row = document.createElement('tr');
    row.dataset.index = g.index;
    row.innerHTML = `<td>${g.index}${g.partial ? '*' : ''}</td><td>${g.plies}</td><td>${g.winner ?? '?'}</td>
                      <td>${g.final_black_captures}/${g.final_white_captures}</td>`;
    row.addEventListener('click', () => loadGame(g.index));
    body.appendChild(row);
  }

  const lastPage = Math.max(1, Math.ceil(state.totalGames / state.pageSize));
  document.getElementById('page-label').textContent = `page ${state.page} / ${lastPage}`;
  document.getElementById('prev-page').disabled = state.page <= 1;
  document.getElementById('next-page').disabled = state.page >= lastPage;
}

document.getElementById('prev-page').addEventListener('click', () => { if (state.page > 1) { state.page--; loadGameList(); } });
document.getElementById('next-page').addEventListener('click', () => { state.page++; loadGameList(); });

document.getElementById('step-first').addEventListener('click', () => { state.ply = 0; renderPly(state.game, state.ply); });
document.getElementById('step-prev').addEventListener('click', () => { state.ply = Math.max(0, state.ply - 1); renderPly(state.game, state.ply); });
document.getElementById('step-next').addEventListener('click', () => { state.ply = Math.min(state.game.plies.length - 1, state.ply + 1); renderPly(state.game, state.ply); });
document.getElementById('step-last').addEventListener('click', () => { state.ply = state.game.plies.length - 1; renderPly(state.game, state.ply); });

loadGameList();
