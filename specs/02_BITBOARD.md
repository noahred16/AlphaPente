## Part 1: Core Implementation

### 1.1 BitBoard Representation

For a 19x19 board (361 positions), we'll use 6 64-bit integers per player.

### 1.2 Efficient Undo with Move Deltas

Move deltas store minimal information needed to reverse any move:
- position: where the stone was placed
- player: who played it
- captured_positions: list of captured stone positions
- capture_count: number of pairs captured

The GameState class maintains:
- Single BitBoard instance
- Move history as list of deltas
- Current player
- Capture counts

Key methods:
- make_move(position) returns MoveDelta
- undo_move() uses last delta to reverse
