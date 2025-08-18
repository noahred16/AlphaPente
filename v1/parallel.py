import multiprocessing as mp
import sqlite3
import time
import random
import os
from settings_loader import (
    BOARD_SIZE,
    TABLE_NAME,
    DB_PATH,
    TOURNAMENT_RULES_ENABLED,
    MODEL,
    CONNECT_N,
    CAPTURES_ENABLED,
)
from db import Database
from models.gomoku_simple_nn import GomokuSimpleNN
from generate import play_game
import torch
import numpy as np

print(f"CPUs on machine: {mp.cpu_count()}")


# def setup_database(db_path):
#     """Create the table if it doesn't exist"""
#     conn = sqlite3.connect(db_path)
#     conn.execute('''CREATE TABLE IF NOT EXISTS messages
#                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                      worker_id INTEGER,
#                      message TEXT,
#                      timestamp REAL)''')
#     conn.commit()
#     conn.close()

# Global queue - will be inherited by worker processes
message_queue = None

def worker_process(worker_id, model_state_dict, num_messages=10):
    """Workers generate messages and put them in the queue"""
    global message_queue
    
    # Initialize the model from shared state dict
    record_count = 0
    model = GomokuSimpleNN(BOARD_SIZE[0])
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
        print(f"Worker {worker_id}: Loaded model from shared state")

    print(f"Worker {worker_id} starting game")
    moves, result = play_game(model)
    print(f"Worker {worker_id} DONE game, with num_moves: {len(moves)}")

    for i, (board, player_captures, opponent_captures, num_moves, policy) in enumerate(
        reversed(moves)
    ):
        if np.all(policy == 0):
            continue
    
        result = result if i % 2 == 0 else -result

        message_queue.put((
            TABLE_NAME,
            board,
            player_captures,
            opponent_captures,
            num_moves,
            policy,
            result,
        ))
        record_count += 1
        # print(f"Worker {worker_id}: Generated message {i+1}")
        if record_count == num_messages:
            print(f"Worker {worker_id} done: Pushed {num_messages} messages")
            return worker_id

    return worker_id

def database_writer(queue, db_path, total_messages):
    """Dedicated process for writing to database"""
    conn = sqlite3.connect(db_path)
    messages_written = 0
    batch = []
    
    print(f"DB Writer: Starting, expecting {total_messages} messages")


    while messages_written < total_messages:
        try:
            # Collect messages into batches
            message_data = queue.get(timeout=1.0)
            batch.append(message_data)

            # print(f"length of batch: {len(batch)}")
            print(f"DB Writer: Got message, batch size: {len(batch)}, total written: {messages_written}/{total_messages}")

            # Write in batches of 10 for efficiency
            if len(batch) >= 10 or queue.empty():

                # TODO write to the correct db!
                # print("yay!")

                # return
                print(f"DB Writer: Would write batch of {len(batch)} messages ({messages_written}/{total_messages} total)")


                # conn.executemany("INSERT INTO messages (worker_id, message, timestamp) VALUES (?, ?, ?)",
                #                batch)
                # conn.commit()
                messages_written += len(batch)
                # print(f"DB Writer: Wrote batch of {len(batch)} messages ({messages_written}/{total_messages} total)")
                batch = []
                
        except:
            # Timeout - write any remaining messages
            if batch:
                # conn.executemany("INSERT INTO messages (worker_id, message, timestamp) VALUES (?, ?, ?)",
                #                batch)
                # conn.commit()
                messages_written += len(batch)
                batch = []
    
    conn.close()

def run_parallel_demo():
    """Main demonstration function"""
    global message_queue
    
    # Load model once in main process
    model_state_dict = None
    checkpoint_path = f"./checkpoints/{MODEL}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_state_dict = checkpoint["model_state"]
        print(f"Loaded model checkpoint from {checkpoint_path}")
    
    # Get SLURM allocation or default to 8
    allocated_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    # Reserve 1 CPU for DB writer, use rest for workers
    num_workers = allocated_cpus - 1
    
    num_messages_per_worker = 10
    total_messages = num_workers * num_messages_per_worker
    
    print(f"CPUs on machine: {mp.cpu_count()}")
    print(f"CPUs allocated by SLURM: {allocated_cpus}")
    print(f"Using {num_workers} workers + 1 DB writer")
    print(f"Each worker will generate {num_messages_per_worker} messages")
    print(f"Total messages: {total_messages}")
    print("-" * 60)
    
    # Initialize the database
    db = Database(DB_PATH, BOARD_SIZE)
    db.setup(TABLE_NAME)
    
    # Create queue - must be created before Pool
    message_queue = mp.Queue()
    
    start_time = time.time()
    
    # Start database writer process
    db_writer = mp.Process(target=database_writer, 
                          args=(message_queue, DB_PATH, total_messages))
    db_writer.start()
    
    # Start workers - pass shared model state dict
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(worker_process,
                             [(i, model_state_dict, num_messages_per_worker) 
                              for i in range(num_workers)])
    
    # Wait for database writer to finish
    db_writer.join()
    
    elapsed_time = time.time() - start_time
    
    # Results
    print("\n" + "-" * 60)
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Rate: {total_messages/elapsed_time:.1f} messages/second")
    
    # Verify
    # conn = sqlite3.connect(DB_PATH)
    # count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    # print(f"Total messages in database: {count}")
    # conn.close()

if __name__ == "__main__":
    print("Starting...")
    run_parallel_demo()