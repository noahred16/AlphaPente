// AlphaPente Web Game JavaScript

class PenteGame {
    constructor() {
        this.gameState = null;
        this.boardSize = 19;
        this.isProcessing = false;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        // Game elements
        this.gameBoard = document.getElementById('gameBoard');
        this.gameStatus = document.getElementById('gameStatus');
        this.humanCaptures = document.getElementById('humanCaptures');
        this.aiCaptures = document.getElementById('aiCaptures');
        this.aiName = document.getElementById('aiName');
        
        // Control elements
        this.newGameBtn = document.getElementById('newGameBtn');
        this.analysisBtn = document.getElementById('analysisBtn');
        this.settingsToggle = document.getElementById('settingsToggle');
        
        // Settings elements
        this.settingsPanel = document.getElementById('settingsPanel');
        this.boardSizeSelect = document.getElementById('boardSize');
        this.aiDifficultySelect = document.getElementById('aiDifficulty');
        this.humanGoesFirstSelect = document.getElementById('humanGoesFirst');
        this.tournamentRuleCheck = document.getElementById('tournamentRule');
        this.capturesToWinInput = document.getElementById('capturesToWin');
        
        // Analysis elements
        this.analysisPanel = document.getElementById('analysisPanel');
        this.moveStats = document.getElementById('moveStats');
        
        // No loading overlay needed
    }
    
    bindEvents() {
        this.newGameBtn.addEventListener('click', () => this.startNewGame());
        this.analysisBtn.addEventListener('click', () => this.toggleAnalysis());
        this.settingsToggle.addEventListener('click', () => this.toggleSettings());
        
        // Auto-update AI name when difficulty changes
        this.aiDifficultySelect.addEventListener('change', () => {
            const difficulty = this.aiDifficultySelect.value;
            this.aiName.textContent = `${difficulty.charAt(0).toUpperCase() + difficulty.slice(1)} AI`;
        });
        
        // Set initial AI name to match default
        this.aiName.textContent = 'Easy AI';
        
        // Auto-start game with default settings when any setting changes
        this.boardSizeSelect.addEventListener('change', () => this.startNewGame());
        this.aiDifficultySelect.addEventListener('change', () => this.startNewGame());
        this.humanGoesFirstSelect.addEventListener('change', () => this.startNewGame());
        this.tournamentRuleCheck.addEventListener('change', () => this.startNewGame());
        this.capturesToWinInput.addEventListener('change', () => this.startNewGame());
    }
    
    async startNewGame() {
        if (this.isProcessing) return;
        
        this.setProcessing(true);
        
        try {
            const settings = {
                board_size: parseInt(this.boardSizeSelect.value),
                captures_to_win: parseInt(this.capturesToWinInput.value),
                tournament_rule: this.tournamentRuleCheck.checked,
                human_goes_first: this.humanGoesFirstSelect.value === 'true',
                ai_difficulty: this.aiDifficultySelect.value
            };
            
            const response = await fetch('/api/new_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.gameState = await response.json();
            this.boardSize = this.gameState.board.length;
            
            this.createBoard();
            this.updateDisplay();
            
            // Enable analysis button
            this.analysisBtn.disabled = false;
            
            // If AI goes first, trigger AI move
            if (!this.gameState.waiting_for_human && !this.gameState.game_over) {
                setTimeout(() => this.makeAIMove(), 500);
            }
            
        } catch (error) {
            console.error('Error starting new game:', error);
            this.showError('Failed to start new game. Please try again.');
        } finally {
            this.setProcessing(false);
        }
    }
    
    createBoard() {
        this.gameBoard.innerHTML = '';
        this.gameBoard.className = `board size-${this.boardSize}`;
        
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = i;
                cell.dataset.col = j;
                
                cell.addEventListener('click', () => this.handleCellClick(i, j));
                
                this.gameBoard.appendChild(cell);
            }
        }
    }
    
    async handleCellClick(row, col) {
        if (this.isProcessing || !this.gameState || !this.gameState.waiting_for_human || this.gameState.game_over) {
            return;
        }
        
        // Let backend handle all move validation
        
        this.setProcessing(true);
        
        try {
            const response = await fetch('/api/make_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ row, col })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                this.showError(errorData.error || 'Failed to make move. Please try again.');
                return;
            }
            
            this.gameState = await response.json();
            this.updateDisplay();
            
            // If it's AI's turn and game isn't over, make AI move
            if (!this.gameState.waiting_for_human && !this.gameState.game_over) {
                setTimeout(() => this.makeAIMove(), 300);
            }
            
        } catch (error) {
            console.error('Error making move:', error);
            this.showError('Failed to make move. Please try again.')
        } finally {
            this.setProcessing(false);
        }
    }
    
    async makeAIMove() {
        if (!this.gameState || this.gameState.waiting_for_human || this.gameState.game_over) {
            return;
        }
        
        // Update status to show AI thinking (no overlay needed)
        this.updateStatus();
        this.setProcessing(true);
        
        try {
            const response = await fetch('/api/ai_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.gameState = await response.json();
            this.updateDisplay();
            
        } catch (error) {
            console.error('Error making AI move:', error);
            this.showError('AI move failed. Please try again.');
        } finally {
            this.setProcessing(false);
        }
    }
    
    updateDisplay() {
        if (!this.gameState) return;
        
        this.updateBoard();
        this.updateStatus();
        this.updateCaptures();
    }
    
    updateBoard() {
        const cells = this.gameBoard.querySelectorAll('.cell');
        
        // Clear previous states
        cells.forEach(cell => {
            cell.innerHTML = '';
            cell.classList.remove('last-move', 'disabled');
        });
        
        // Place stones
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const cellIndex = i * this.boardSize + j;
                const cell = cells[cellIndex];
                const value = this.gameState.board[i][j];
                
                if (value !== 0) {
                    const stone = document.createElement('div');
                    stone.className = `stone ${value === 1 ? 'black' : 'white'}`;
                    cell.appendChild(stone);
                }
                
                // Highlight last move
                if (this.gameState.move_history.length > 0) {
                    const lastMove = this.gameState.move_history[this.gameState.move_history.length - 1];
                    if (lastMove[0] === i && lastMove[1] === j) {
                        cell.classList.add('last-move');
                    }
                }
                
                // Disable cells when not human's turn or game over
                if (!this.gameState.waiting_for_human || this.gameState.game_over || this.isProcessing) {
                    cell.classList.add('disabled');
                }
            }
        }
    }
    
    updateStatus() {
        const status = this.gameStatus;
        status.className = 'game-status';
        
        if (this.gameState.game_over) {
            status.classList.add('game-over');
            
            if (this.gameState.winner === this.gameState.human_player_id) {
                status.textContent = '🎉 You Win!';
                status.classList.add('you-win');
            } else if (this.gameState.winner === this.gameState.ai_player_id) {
                status.textContent = '🤖 AI Wins!';
            } else {
                status.textContent = '🤝 Draw!';
            }
        } else if (this.gameState.waiting_for_human) {
            status.textContent = '👤 Your Turn';
            status.classList.add('your-turn');
        } else {
            status.textContent = '🤖 AI Thinking...';
            status.classList.add('ai-thinking');
        }
    }
    
    updateCaptures() {
        this.humanCaptures.textContent = this.gameState.captures[this.gameState.human_player_id] || 0;
        this.aiCaptures.textContent = this.gameState.captures[this.gameState.ai_player_id] || 0;
    }
    
    async toggleAnalysis() {
        if (this.analysisPanel.style.display === 'none') {
            await this.loadAnalysis();
            this.analysisPanel.style.display = 'block';
            this.analysisBtn.textContent = 'Hide Analysis';
        } else {
            this.analysisPanel.style.display = 'none';
            this.analysisBtn.textContent = 'Show Analysis';
        }
    }
    
    async loadAnalysis() {
        if (!this.gameState || this.gameState.game_over) {
            this.moveStats.innerHTML = '<p>No analysis available.</p>';
            return;
        }
        
        try {
            const response = await fetch('/api/move_statistics');
            const data = await response.json();
            
            if (data.statistics && data.statistics.length > 0) {
                this.displayAnalysis(data.statistics);
            } else {
                this.moveStats.innerHTML = '<p>Analysis not available for this position.</p>';
            }
        } catch (error) {
            console.error('Error loading analysis:', error);
            this.moveStats.innerHTML = '<p>Failed to load analysis.</p>';
        }
    }
    
    displayAnalysis(statistics) {
        // Sort by visits (most explored first)
        statistics.sort((a, b) => b[1] - a[1]);
        
        let html = '<h5>Top AI Moves Analysis:</h5>';
        
        statistics.slice(0, 10).forEach((stat, index) => {
            const [move, visits, winRate] = stat;
            const [row, col] = move;
            const percentage = (winRate * 100).toFixed(1);
            
            const moveClass = index === 0 ? 'move-stat best' : 'move-stat';
            
            html += `
                <div class="${moveClass}">
                    <span>Move: (${row}, ${col})</span>
                    <span>${visits} visits, ${percentage}% win rate</span>
                </div>
            `;
        });
        
        this.moveStats.innerHTML = html;
    }
    
    toggleSettings() {
        const isVisible = this.settingsPanel.style.display !== 'none';
        this.settingsPanel.style.display = isVisible ? 'none' : 'block';
        this.settingsToggle.textContent = isVisible ? 'Show Settings' : 'Hide Settings';
    }
    
    setProcessing(processing) {
        this.isProcessing = processing;
        
        // Disable/enable controls
        this.newGameBtn.disabled = processing;
        this.analysisBtn.disabled = processing || !this.gameState;
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a proper modal
        alert(message);
    }
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.penteGame = new PenteGame();
    
    // Auto-start game with default settings
    setTimeout(() => {
        window.penteGame.startNewGame();
    }, 100);
});