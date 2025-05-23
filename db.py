import numpy as np
import sqlite3
import os


class Database:
    def __init__(self, db_path, board_size):
        self.db_path = db_path
        self.board_size = board_size

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
        conn.commit()
        conn.close()

    def fetch_collection(self, table_name, n, columns="*"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {columns}
            FROM {table_name}
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        )

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
