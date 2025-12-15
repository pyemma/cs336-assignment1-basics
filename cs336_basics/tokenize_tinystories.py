import argparse
import multiprocessing as mp
import time

import numpy as np

from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries

def tokenize_chunk(
    input_path: str,
    start_idx: int,
    end_idx: int,
    vocab_path: str,
    merges_path: str,
    queue: mp.Queue,
):
    """
    Tokenize a chunk of the input file and pass the output to the queue
    """
    start = time.time()
    with open(input_path, "rb") as f:
        f.seek(start_idx)
        chunk = f.read(end_idx - start_idx).decode("utf-8", errors="ignore")
    
    print(f"Tokenizing chunk {start_idx} to {end_idx}")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    ids = tokenizer.encode(chunk)
    queue.put(ids)
    end = time.time()
    print(f"Time taken: {end - start} seconds for chunk {start_idx} to {end_idx}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "valid"])

    args = parser.parse_args()
    print(args)
    input_path = f"/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-{args.mode}.txt"
    output_path = f"/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/data/tinystories-ids-{args.mode}.bin"
    vocab_path = "/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/result/vocab-tinystories.pkl"
    merges_path = "/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/result/merges-tinystories.pkl"

    num_processes = 16
    # Use Manager().Queue() for better compatibility with spawn method on macOS
    manager = mp.Manager()
    queue = manager.Queue()  # use a queue style solution to collect result

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    processes = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        process = mp.Process(target=tokenize_chunk, args=(input_path, start_idx, end_idx, vocab_path, merges_path, queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    all_ids = []
    for _ in range(len(processes)):
        ids = queue.get()
        all_ids.extend(ids)
    
    # save the ids to a file
    all_ids = np.array(all_ids, dtype=np.uint16)  # the total vocab size is 10000 which is less than 16 bits
    all_ids.tofile(output_path)



