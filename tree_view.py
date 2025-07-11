import math
import graphviz
import io
from contextlib import redirect_stdout
from game import pretty_print  # Import pretty_print from game.py
from mcts import MCTSNode  # Import MCTSNode from mcts.py
from collections import deque

NODE_LIMIT = 400  # Maximum number of nodes to display in the tree visualization


# a method that takes the root of the MCTS tree and generates a visualization of the tree
def tree_visualization(root: MCTSNode, exploration_constant: float = 1.0):
    from collections import deque

    # Create a graph with a larger node size to accommodate the game display
    graph = graphviz.Digraph(format="png")
    graph.attr(
        "node", shape="box", fontname="Courier New"
    )  # Monospace font for game display

    # Use breadth-first traversal
    queue = deque([root])
    node_count = 0

    while queue and node_count < NODE_LIMIT:
        node = queue.popleft()
        node_id = str(id(node))
        node_count += 1

        # Capture the output of pretty_print()
        output = io.StringIO()
        with redirect_stdout(output):
            # Flip the board for display
            flipped_board = node.board * node.turn * -1
            pretty_print(flipped_board, move=node.prev_move, ASCII=True)

        game_display = output.getvalue()

        ucb = round(node.get_puct(), 2) if node.puct is not None else "N/A"

        # Safely handle None values for total_value and num_visits
        total_value = node.total_value if node.total_value is not None else 0.0
        num_visits = (
            node.num_visits if node.num_visits > 0 else 1
        )  # Avoid division by zero
        exploration = node.exploration if node.exploration is not None else 0
        exploitation = node.exploitation if node.exploitation is not None else 0
        value = node.value if node.value is not None else 0.0

        # Format the label with statistics and game state
        label = (
            f"Move: {node.prev_move if node.prev_move is not None else 'Root'}\\n"
            f"Total Value: {total_value:.2f}\\n"
            f"Visits: {node.num_visits}\\n"
            f"Avg Value: {total_value / num_visits:.2f}\\n"
            f"Exploration: {exploration:.2f}\\n"
            f"Exploitation: {exploitation:.2f}\\n"
            f"PUCT: {ucb}\\n"
            f"is_terminal: {node.is_terminal}\\n"
            f"value: {value:.2f}\\n"
            f"num_moves: {node.num_moves}\\n"
            f"Player: {'X' if node.turn == 1 else 'O'}\\n"
        )
        label += game_display.replace("\n", "\\n")

        # Apply different styling based on node type and turn
        if node.is_terminal:
            # Terminal nodes: check for tie (value == 0), otherwise color based on turn
            if value == 0.0:
                # Tie - yellow color
                fill_color = "yellow"
            else:
                # Win/loss - red if node.turn == 1, green otherwise
                fill_color = "lightcoral" if node.turn == 1 else "lightgreen"
            graph.node(
                node_id,
                label=label,
                style="bold,filled",
                penwidth="3",
                fillcolor=fill_color,
                color="black",
            )
        else:
            # Non-terminal nodes: light grey if node.turn != 1, normal otherwise
            if node.turn == 1:
                graph.node(node_id, label=label)
            else:
                graph.node(node_id, label=label, style="filled", fillcolor="lightgrey")

        if node.parent is not None:
            graph.edge(str(id(node.parent)), node_id, label=str(node.prev_move))

        # Add children to the queue for breadth-first traversal
        if node_count < NODE_LIMIT:
            # Sort children by move coordinates (row, column) in ascending order
            # sorted_children = sorted(
            #     node.children.items(), key=lambda x: (x[0][0], x[0][1])
            # )
            for child in node.children:
                queue.append(child)

    file_name = "mcts_tree"
    graph.render(file_name, cleanup=True)
    print(f"Tree visualization saved as {file_name}.png")
