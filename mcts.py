import numpy as np
import math
import random
import torch
from settings_loader import (
    BOARD_SIZE,
    CAPTURES_ENABLED,
    TOURNAMENT_RULES_ENABLED,
    CONNECT_N,
    PUCT_EXPLORATION_CONSTANT,
)
from game import (
    reset_game,
    get_legal_moves,
    make_move,
    is_game_over,
    calculate_score,
    pretty_print,
)
from models.gomoku_simple_nn import create_board_state


class MCTSNode:
    def __init__(
        self,
        board,
        policy_prior,
        player_captures=0,
        opponent_captures=0,
        num_moves=0,
        prev_move=None,  # we don't need move in here but its an optimzaiton for the is_game_over
        parent=None,
    ):
        self.board = board.copy()
        self.board = np.array(self.board, dtype=np.float32)  # Ensure board is float32
        self.board *= -1  # Flip the board for the current player

        self.prev_move = prev_move
        make_move(prev_move, self.board, 1) if prev_move is not None else None

        # make move
        self.policy_prior = policy_prior
        self.player_captures = player_captures
        self.opponent_captures = opponent_captures
        self.num_moves = num_moves
        self.parent = parent
        self.children = {}
        self.num_visits = 0
        self.is_terminal = is_game_over(
            prev_move, self.board, player_captures, CONNECT_N, num_moves
        )
        self.untried_moves = (
            get_legal_moves(self.board, num_moves, TOURNAMENT_RULES_ENABLED)
            if not self.is_terminal
            else []
        )
        self.value = (
            calculate_score(num_moves, BOARD_SIZE, CAPTURES_ENABLED)
            if self.is_terminal
            else None
        )
        self.total_value = 0  # self.value if self.is_terminal else 0
        self.turn = 1
        if parent:
            self.turn = -parent.turn

        self.puct = None
        self.exploration = None
        self.exploitation = None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def get_puct(self):
        """
        PUCT formula:
            Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        """
        if self.num_visits == 0:
            score = float("inf")
            self.ucb_score = score
            return score

        exploitation = self.total_value / self.num_visits  #  * self.turn

        c_puct = PUCT_EXPLORATION_CONSTANT
        # c_puct = 1.5

        # Convert 2D move coordinates to 1D index for policy_prior
        if self.prev_move is not None:
            # row, col = self.prev_move
            # move_index = row * BOARD_SIZE[0] + col
            prior = self.policy_prior
            # prior = 1
        else:
            # This shouldn't happen since root node doesn't use PUCT
            prior = 1.0 / (BOARD_SIZE[0] * BOARD_SIZE[0])

        exploration = (
            c_puct * prior * math.sqrt(self.parent.num_visits) / (1 + self.num_visits)
        )

        self.exploration = exploration
        self.exploitation = exploitation

        score = exploitation + exploration
        return score

    def best_child(self, greedy=False):
        best_node = None
        best_value = float("-inf")

        for move, child in self.children.items():
            if greedy:
                # value = child.total_value / child.num_visits if child.num_visits > 0 else 0
                value = child.num_visits
            else:
                child.puct = child.get_puct()
                value = child.puct

            if value > best_value:
                best_value = value
                best_node = child
        return best_node


class MCTS:
    def __init__(self, model, simulations=100, exploration_constant=1.0, random=0.0):
        """
        Initialize the MCTS with the game state and model.
        :param game: The game instance.
        :param model: The neural network model for policy and value predictions.
        :param simulations: Number of simulations to run per move.
        """
        self.model = model
        self.simulations = simulations
        self.exploration_constant = exploration_constant  # TODO i dunno

        # TODO: Initialize any additional data structures needed for MCTS.

        self.random = random
        self.samples = []

        self.root = None

    def best_move(self, board, player_captures, opponent_captures):
        root = self.run(board, player_captures, opponent_captures)
        if root is None:
            return None

        # using self.random check if we should chose a random legal move

        if self.random != 0.0 and random.random() < self.random:
            legal_moves = get_legal_moves(
                board, root.num_moves, TOURNAMENT_RULES_ENABLED
            )
            if legal_moves:
                return random.choice(legal_moves), np.zeros(
                    (BOARD_SIZE[0], BOARD_SIZE[1]), dtype=float
                )
            else:
                raise ValueError("No legal moves available for random selection.")
            best = random.choice(root.children.values())

        best = root.best_child(greedy=True)

        # TODO
        # policy is determined by using the move nums (num_visits)
        # np.array(dtype=float) of size BOARD_SIZE[0] * BOARD_SIZE[1]

        moves_visits = []
        # loop through the moves from the root and
        for move, child in root.children.items():
            if child.num_visits > 0:
                moves_visits.append((move, child.num_visits))

        # policy
        policy = np.zeros((BOARD_SIZE[0], BOARD_SIZE[1]), dtype=float)

        # scale so the sum is 1
        total_visits = sum(visits for _, visits in moves_visits)
        if total_visits > 0:
            for move, visits in moves_visits:
                row, col = move
                policy[row][col] = visits / total_visits

        return best.prev_move, policy

    def run(self, board, player_captures, opponent_captures):
        """
        Run MCTS for the specified number of simulations.
        :return: The best move based on the search.
        """
        num_moves = sum(1 for row in board for cell in row if cell != 0)

        root_node = MCTSNode(
            board=board,
            policy_prior=None,
            player_captures=player_captures,
            opponent_captures=opponent_captures,
            num_moves=num_moves,
        )
        self.root = root_node
        for _ in range(self.simulations):
            # 1.) SELECTION
            leaf = self.selection(root_node)

            # 2.) EXPANSION
            child = self.expansion(leaf)

            # 3.) SIMULATION
            result = self.rollout(child)

            # 4.) BACKPROPAGATION
            self.backpropagation(child, result)

        return self.root

    # .get_best_move()

    def predict_policy_and_value(self, board, player_captures, opponent_captures):
        board_input = torch.tensor(board, dtype=torch.float32)
        board_state = create_board_state(board_input, 1)  # TODO check assumption
        with torch.no_grad():
            move_probs, board_value = self.model.predict(board_state)

            # Convert move_probs to policy priors
            policy = move_probs.numpy()
            value = board_value.item()

            # player_captures_input = torch.tensor(player_captures, dtype=torch.float32)
            # opponent_captures_input = torch.tensor(
            #     opponent_captures, dtype=torch.float32
            # )

            # policy, value = self.model(
            #     board_input.unsqueeze(0),
            #     player_captures_input.unsqueeze(0),
            #     opponent_captures_input.unsqueeze(0),
            # )

            # policy = policy.squeeze(0).numpy()
            # value = value.item()
        return policy, value

    def get_policy_priors(self, board, player_captures, opponent_captures):
        policy, _ = self.predict_policy_and_value(
            board, player_captures, opponent_captures
        )
        return policy

    def get_value(self, board, player_captures, opponent_captures):
        # return 0
        _, value = self.predict_policy_and_value(
            board, player_captures, opponent_captures
        )
        return value

    def selection(self, node: MCTSNode):
        """
        Select the best child node based on the PUCT formula.
        :param node: The current node.
        :return: The selected child node.
        """
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child()
        return node

    def expansion(self, node: MCTSNode):
        """
        Expand the current node by adding all possible child nodes.
        :param node: The current node.
        """
        if node.is_terminal:
            return node

        # should we use the model to get policy priors here?
        # action = random.choice(node.untried_moves)
        # node.untried_moves.remove(action)

        # TODO flip board
        # next_turn_board = node.board

        # TODO super key. board seems the same between mveos.
        # board_copy = node.board.copy()
        # make the move on the board copy
        # board_copy = make_move(board_copy, action, node.turn, TOURNAMENT_RULES_ENABLED)

        # TODO apply the move to the board?
        # for each move in untried_moves:
        policy_priors = self.get_policy_priors(
            board=node.board,
            player_captures=node.player_captures,
            opponent_captures=node.opponent_captures,
        )
        for move in node.untried_moves:
            move_index = move[0] * BOARD_SIZE[0] + move[1]
            child_node = MCTSNode(
                prev_move=move,
                policy_prior=policy_priors[move_index],
                board=node.board,
                player_captures=node.player_captures,
                opponent_captures=node.opponent_captures,
                num_moves=node.num_moves + 1,
                parent=node,
            )
            node.children[move] = child_node
            node.untried_moves.remove(move)

        return child_node

    def rollout(self, node: MCTSNode):
        """
        Simulate a game from the current node to a terminal state.
        :param node: The current node.
        :return: The value of the terminal state.
        """
        if node.is_terminal:
            return node.value  # TODO account for turn?

        value = self.get_value(
            node.board,
            node.player_captures,
            node.opponent_captures,
        )
        return value

    def backpropagation(self, node: MCTSNode, value: float):
        """
        Backpropagate the simulation result up the tree.
        :param node: The current node.
        :param value: The value to backpropagate.
        """
        player = node.turn
        while node is not None:
            node.num_visits += 1
            # account for turn
            is_opp = -1 if player != node.turn else 1
            node.total_value += value * is_opp
            node = node.parent

    # TODO DO NOT DELETE
    # def collect_samples(self):
    # using the root
    # loop through the moves and add the
    # move probabilities come from num_visits and the value comes from the value.
    # only collect samples from the greedy path?
    # it may make sense to collect using unique starting points. But we'll always use the root for collcetion
