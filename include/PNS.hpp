/*
Proof Number Search (PNS)
Each iteration starts at the root until the proof number or disproof number reaches 0

Win" is relative to the player who made the first move (the Root player).

Open Q's:
- If a node has two parents, and those parents share the same parent, when we backprop the node, wouldnt it get counted twice in the grandparent?
Yes. Ignore it or use a solution such as "PNS^2" or "GHI". 
- Transposition table stored in RAM or disk or hybrid? 
A high-speed LRU (Least Recently Used) Cache in RAM and a Persistent Key-Value Store on disk. 
Use a packed struct to save bytes. #pragma pack(push, 1) to prevent compiler from adding padding. 


Stages:
1.) Selection 
- traverse to an unexpanded leaf node.
- At an OR Node (your turn), selects the child with the smallest Proof Number. We want to prove a win? 
- At an AND Node (opps turn), selects the child with the smallest Disproof number. Opp wants a win? 

2.) Expansion
- Generate all possible legal moves from that state
- Check if any are terminal **
- Nodes get pn=1 dn=1. Wins pn=0 dn=inf. Loss/Draw pn=inf dn=0

3.) Backpropagation
- OR Nodes (your turn): pn = min(pn of children) dn = sum(dn of children)
- AND nodes (opps turn): pn = sum(pn of children) dn = min(dn of children)

4.) Termination Check
- if root pn or dn = 0, end. pn=0, first player wins. dn = 0, lose. 
*/


// Plan. We can use a MCTS solver policy value NN to efficiently do PNS. Policy for proofs and Value for disproof. 