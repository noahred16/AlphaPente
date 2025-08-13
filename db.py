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
