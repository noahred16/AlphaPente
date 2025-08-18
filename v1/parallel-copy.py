import multiprocessing as mp
import sqlite3
import time
import random
import os

print(f"CPUs on machine: {mp.cpu_count()}")


def setup_database(db_path):
    """Create the table if it doesn't exist"""
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS messages
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     worker_id INTEGER,
                     message TEXT,
                     timestamp REAL)''')
    conn.commit()
    conn.close()

# Global queue - will be inherited by worker processes
message_queue = None

def worker_process(worker_id, num_messages=10):
    """Workers generate messages and put them in the queue"""
    global message_queue
    
    for i in range(num_messages):
        message = f"Hello world from worker {worker_id}, message {i+1}"
        timestamp = time.time()
        
        # Simulate some work (like your MCTS game generation)
        time.sleep(random.uniform(2, 5))
        
        message_queue.put((worker_id, message, timestamp))
        print(f"Worker {worker_id}: Generated message {i+1}")
    
    return worker_id

def database_writer(queue, db_path, total_messages):
    """Dedicated process for writing to database"""
    conn = sqlite3.connect(db_path)
    messages_written = 0
    batch = []
    
    while messages_written < total_messages:
        try:
            # Collect messages into batches
            message_data = queue.get(timeout=1.0)
            batch.append(message_data)
            
            # Write in batches of 10 for efficiency
            if len(batch) >= 10 or queue.empty():
                conn.executemany("INSERT INTO messages (worker_id, message, timestamp) VALUES (?, ?, ?)",
                               batch)
                conn.commit()
                messages_written += len(batch)
                print(f"DB Writer: Wrote batch of {len(batch)} messages ({messages_written}/{total_messages} total)")
                batch = []
                
        except:
            # Timeout - write any remaining messages
            if batch:
                conn.executemany("INSERT INTO messages (worker_id, message, timestamp) VALUES (?, ?, ?)",
                               batch)
                conn.commit()
                messages_written += len(batch)
                batch = []
    
    conn.close()

def run_parallel_demo():
    """Main demonstration function"""
    global message_queue
    
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
    
    # Setup
    db_path = "parallel_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    setup_database(db_path)
    
    # Create queue - must be created before Pool
    message_queue = mp.Queue()
    
    start_time = time.time()
    
    # Start database writer process
    db_writer = mp.Process(target=database_writer, 
                          args=(message_queue, db_path, total_messages))
    db_writer.start()
    
    # Start workers - queue is inherited, not passed
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(worker_process,
                             [(i, num_messages_per_worker) 
                              for i in range(num_workers)])
    
    # Wait for database writer to finish
    db_writer.join()
    
    elapsed_time = time.time() - start_time
    
    # Results
    print("\n" + "-" * 60)
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Rate: {total_messages/elapsed_time:.1f} messages/second")
    
    # Verify
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Total messages in database: {count}")
    conn.close()

if __name__ == "__main__":
    print("Starting...")
    run_parallel_demo()