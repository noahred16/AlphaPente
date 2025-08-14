"""Database Interface for AlphaPente Training Data Storage

This module provides SQLite-based storage for training data generated during self-play games.
It handles serialization/deserialization of game states and supports multiple sampling strategies.

Key Features:
- Efficient storage of board states, policies, and game results
- Multiple sampling strategies for training data retrieval
- Binary serialization of numpy arrays for space efficiency
- Automatic database and table setup
- Transaction management for data integrity

Data Schema:
    - id: Auto-incrementing primary key
    - board: Serialized board state (7x7 numpy array)
    - player_captures: Number of pieces captured by current player
    - opponent_captures: Number of pieces captured by opponent
    - num_moves: Move number in the game
    - policy: Serialized policy distribution (7x7 numpy array)
    - value: Game result from current player's perspective (-1 to 1)

Sampling Strategies:
    - 'recent': Most recent records (ORDER BY id DESC)
    - 'random': Random sampling across all records
    - 'top_random': Mix of 70% recent + 30% random older records

Usage:
    db = Database('game_data.db', (7, 7))
    db.setup('training_data')
    db.store('training_data', board, p_caps, o_caps, moves, policy, value)
    records = db.fetch_collection('training_data', 1000, strategy='top_random')
"""

import numpy as np
import sqlite3
import os


class Database:
    def __init__(self, db_path, board_size):
        self.db_path = db_path
        self.board_size = board_size
        self.setup_empty_db()

    def setup_empty_db(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            conn.close()

    def setup(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board BLOB NOT NULL,
                player_captures INTEGER NOT NULL,
                opponent_captures INTEGER NOT NULL,
                num_moves INTEGER NOT NULL,
                policy BLOB,
                value REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def store(
        self,
        table_name,
        board,
        player_captures,
        opponent_captures,
        num_moves,
        policy,
        value,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        serialized_board = np.array(board, dtype=int).tobytes()
        serialized_policy = np.array(policy, dtype=float).tobytes()
        cursor.execute(
            f"""
            INSERT INTO {table_name} (board, player_captures, opponent_captures, num_moves, policy, value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                serialized_board,
                player_captures,
                opponent_captures,
                num_moves,
                serialized_policy,
                value,
            ),
        )

        if cursor.rowcount == 0:
            raise Exception("Failed to insert data into the database.")

        conn.commit()
        conn.close()

    def fetch_collection(self, table_name, n, columns="*", strategy="recent"):
        """
        Fetch training records using specified sampling strategy.
        
        Args:
            table_name: Name of the database table
            n: Number of records to fetch
            columns: Columns to select (default: all)
            strategy: Sampling strategy ('recent', 'random', 'top_random')
            
        Returns:
            List of database records as tuples
            
        Strategy Details:
            - 'recent': Returns most recent n records (fastest, may overfit)
            - 'random': Returns random n records from entire dataset
            - 'top_random': Returns 70% from recent records + 30% random from older
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if strategy == "recent":
            # Get most recent records
            cursor.execute(
                f"""
                SELECT {columns}
                FROM {table_name}
                ORDER BY id DESC
                LIMIT ?
                """,
                (n,),
            )
        elif strategy == "random":
            # Get completely random records
            cursor.execute(
                f"""
                SELECT {columns}
                FROM {table_name}
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (n,),
            )
        elif strategy == "top_random":
            # Get mix of top records and random older records
            top_n = int(n * 0.7)  # 70% from recent
            random_n = n - top_n  # 30% from older records
            
            # Get top records
            cursor.execute(
                f"""
                SELECT {columns}
                FROM {table_name}
                ORDER BY id DESC
                LIMIT ?
                """,
                (top_n,),
            )
            top_results = cursor.fetchall()
            
            # Get random records from the rest
            cursor.execute(
                f"""
                SELECT {columns}
                FROM {table_name}
                WHERE id NOT IN (
                    SELECT id FROM {table_name}
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (top_n, random_n),
            )
            random_results = cursor.fetchall()
            
            results = top_results + random_results
            conn.close()
            return results
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}. Available: 'recent', 'random', 'top_random'")

        results = cursor.fetchall()
        conn.close()
        return results

    def get_total(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = cursor.fetchone()[0]

        conn.close()
        return total

    def show_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        conn.close()
        return [table[0] for table in tables]

    def get_count(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def serialize(self, board, dtype=int):
        return np.array(board, dtype=dtype).tobytes()

    def deserialize(self, blob, dtype=int):
        return np.frombuffer(blob, dtype=dtype).reshape(self.board_size)

    def decode_row(self, row):
        (
            id,
            board_blob,
            player_captures,
            opponent_captures,
            num_moves,
            policy_blob,
            value,
        ) = row
        board = self.deserialize(board_blob)
        policy = self.deserialize(policy_blob, dtype=float)
        return id, board, player_captures, opponent_captures, num_moves, policy, value

    def delete_by_num_moves(self, table_name, num_moves):
        """Delete all records where num_moves equals the specified value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            DELETE FROM {table_name}
            WHERE num_moves = ?
            """,
            (num_moves,),
        )
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        return affected_rows

    def clear_table(self, table_name):
        """Clear all records from the specified table."""
        if not table_name:
            raise ValueError("Table name cannot be empty.")
        input_val = input(
            f"Are you sure you want to clear the table '{table_name}'? (yes/no): "
        )
        if input_val.lower() not in ["yes", "y"]:
            print("Operation cancelled.")
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Cleared {affected_rows} rows from table '{table_name}'.")
        return affected_rows

    def review_last_game(self, table_name):
        # get rows where num_moves = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT *
            FROM {table_name}
            WHERE num_moves = 0
            ORDER BY id DESC
            LIMIT 2
            """
        )
        starting_points = cursor.fetchall()
        if len(starting_points) != 2:
            print("No starting points found in the database.")
            return []
        # decode
        data = []
        for start in starting_points:
            data.append(self.decode_row(start))

        start_id = starting_points[0][0]
        second_starting_point_id = starting_points[1][0]

        # fetch all rows with id > second_starting_point_id
        cursor.execute(
            f"""
            SELECT *
            FROM {table_name}
            WHERE id > ?
            """,
            (second_starting_point_id,),
        )
        review_data = cursor.fetchall()
        conn.close()
        return review_data
