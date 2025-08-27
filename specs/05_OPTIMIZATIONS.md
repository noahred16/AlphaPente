## Part 2: Advanced Optimizations

### 2.1 Progressive Widening Strategy
- First visit: 10-15 moves at distance 1
- After 100 visits: 20-30 moves at distance 1-2
- After 1000 visits: 40-50 moves at distance 1-3
- After 10000 visits: Consider all reasonable moves

### 2.2 Node Pool for Memory Management
- Pre-allocate node objects
- Reuse instead of creating new
- Reduces allocation overhead

### 2.3 Early Termination in Rollouts
- Check if position clearly won/lost
- Stop rollout early if decided
- Saves simulation time

### 2.4 Smart Time Management
- Monitor best vs second-best visits
- Stop early if one move dominates
- Typically when best > 3x second

### 2.5 RAVE/AMAF (Optional)
- Track move success regardless of when played
- Combine with UCT values
- Beta parameter decreases with visits
