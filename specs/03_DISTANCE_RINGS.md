### 1.3 Precomputed Distance Rings

Initialize at startup:
- Create 361x19 structure (each position to all distances)
- Calculate each pair distance only ONCE
- Store in both positions' lookup tables
- Access is O(1) during search

Structure shape:
- rings[position][distance] = [list of positions at that exact distance]
- Example: rings[180][1] = 8 adjacent positions to center

### 1.4 Fast Move Generation with Distance Ordering

get_ordered_moves(state):
1. Get all stone positions from move history
2. For each distance (1, 2, 3...):
   - For each stone:
     - Look up positions at this distance
     - Add unoccupied ones to result
3. First encounter = minimum distance
4. Return positions ordered by distance

Progressive widening based on node visits:
- <10 visits: distance 1, 15 moves
- <100 visits: distance 2, 30 moves  
- <1000 visits: distance 3, 50 moves
- 1000+ visits: distance 5, 80 moves
