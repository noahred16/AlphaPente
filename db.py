import numpy as np
import sqlite3
import os


class Database:
    def __init__(self, db_path):
        self.db_path = db_path

    def setup(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pente_games (
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
        self, board, player_captures, opponent_captures, num_moves, policy, value
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        serialized_board = np.array(board, dtype=int).tobytes()
        serialized_policy = np.array(policy, dtype=float).tobytes()
        cursor.execute(
            """
            INSERT INTO pente_games (board, player_captures, opponent_captures, num_moves, policy, value)
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


def fetch_collection(db_path, n):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    SELECT id, board, player_captures, opponent_captures, num_moves, policy, value
    FROM pente_games
    ORDER BY id DESC
    LIMIT ?
    """,
        (n,),
    )

    results = cursor.fetchall()
    conn.close()
    if not results:
        return []
    return results


def get_total(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pente_games")
    total = cursor.fetchone()[0]

    conn.close()
    return total


def serialize(board, dtype=int):
    return np.array(board, dtype=dtype).tobytes()


def deserialize(blob, dtype=int):
    return np.frombuffer(blob, dtype=dtype).reshape((19, 19))
