"""
A simple script to train BPE tokenizer

1. Pre-tokenization using multiprocessing, obtain `dict[tuple[bytes], int]`
2. Initialize the vocab with the default 256 characters and special tokens
3. Run iterative loop to compute BPE merge and update teh vocab until hit maximum tokens 
"""
from collections import defaultdict
import multiprocessing as mp
import os
import regex as re
from sortedcontainers import SortedDict
from typing import BinaryIO

from cs336_basics.constants import PAT

# function copied over from pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenization(
    input_path: str,
    start_idx: int,
    end_idx: int,
    special_tokens: list[str],
    queue: mp.Queue
) -> dict[tuple[bytes], int]:
    """
        The pretokeinzation step, which could be run in a multiprocessing step
    """
    chunk = ""
    with open(input_path, "rb") as f:
        f.seek(start_idx)
        chunk = f.read(end_idx - start_idx).decode("utf-8", errors="ignore")

    # remove special tokens
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    docs = re.split("|".join(escaped_special_tokens), chunk)

    # count
    counter = defaultdict(int)
    for idx, doc in enumerate(docs):
        matches_iterator = re.finditer(PAT, doc)
        for match in matches_iterator:
            s = match.group()
            # encode the tokens to bytes, convert to tuple of bytes
            counter[tuple(bytes([b]) for b in s.encode("utf-8"))] += 1
        
        if (idx + 1) % 500 == 0:
            print(f"start idx {start_idx} computed {idx + 1} documents...")
    
    # return the counts
    print(f"start idx {start_idx} finished pretokenization...")
    queue.put(counter)


def update_counter(pair, count, bpe_pairs, counter_index):
    """
    A simple function to update the bpe pair count information
    """
    current_count = bpe_pairs[pair]
    if current_count > 0:
        counter_index[current_count].remove(pair)
        if not counter_index[current_count]:
            del counter_index[current_count]
    updated_count = current_count + count
    if updated_count > 0:
        if updated_count not in counter_index:
            counter_index[updated_count] = set()
        counter_index[updated_count].add(pair)
        bpe_pairs[pair] = updated_count
    else:
        del bpe_pairs[pair]

def merge_token(token, bpe_pair, indices):
    merged_token = []
    merged_pair = (bpe_pair[0] + bpe_pair[1], )
    merged_token = tuple()
    for idx, i in enumerate(indices):
        if idx == 0:
            merged_token += token[:i]
            merged_token += merged_pair
        elif idx > 0 and (indices[idx] - indices[idx-1]) == 1:
            merged_token += (token[i], )  # no merge
        else:
            merged_token += token[indices[idx-1]+2:i]
            merged_token += merged_pair
    # merge the remaining part
    merged_token += token[indices[-1]+2:]
    return merged_token

def update_bpe_pairs(token, merged_token, original_token, token_count, bpe_pairs, counter_index, pair_to_token):
    """
    A naive implementation to compute the new token after merge and update the bpe pairs
    """
    orig_count = defaultdict(int)
    for l, r in zip(token[:-1], token[1:]):
        orig_count[(l, r)] += token_count

    new_count = defaultdict(int)
    if len(merged_token) > 1:  # still pairs available
        for l, r in zip(merged_token[:-1], merged_token[1:]):
            new_count[(l, r)] += token_count
    
    # start to compute update
    for k, v in orig_count.items():
        update_counter(k, -v, bpe_pairs, counter_index)
        pair_to_token[k].remove(original_token)
    for k, v in new_count.items():
        update_counter(k, v, bpe_pairs, counter_index)
        pair_to_token[k].add(original_token)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = 8
    # Use Manager().Queue() for better compatibility with spawn method on macOS
    manager = mp.Manager()
    queue = manager.Queue()  # use a queue style solution to collect result

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # the number of chunk might be smaller then num_processes
    processes = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        process = mp.Process(target=pretokenization, args=(input_path, start_idx, end_idx, special_tokens, queue))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

    pretoken_counter = defaultdict(int)
    for _ in range(len(processes)):
        counter = queue.get()
        for key, value in counter.items():
            pretoken_counter[key] += value
    
    print(f"Total number of pretoken: {len(pretoken_counter)}")

    vocab = {i: bytes([i]) for i in range(256)}  # the initial vocab
    for special_token in special_tokens:  # assign id to special tokens
        vocab[len(vocab)] = special_token.encode("utf-8")

    bpe_pairs_merged = []  # the bpe pairs that get merged
    bpe_pairs = defaultdict(int)  # the counter of bpe pairs
    counter_index = SortedDict()
    pair_to_token = defaultdict(set)  # from token to bpe back tracking
    # construct the initial bpe pair counts
    for key, value in pretoken_counter.items():
        for l, r in zip(key[:-1], key[1:]):
            bpe_pairs[(l, r)] += value
            pair_to_token[(l, r)].add(key)
    # keep a record of the original pretoken and the merged pretoken
    original_to_merged = {
        key: key for key in pretoken_counter
    }

    # index the counts
    for pair, count in bpe_pairs.items():
        if count not in counter_index:
            counter_index[count] = set()
        counter_index[count].add(pair)
    
    while len(vocab) < vocab_size:  # loop until we merge sufficient number of tokens
        # retrieval the current largest count and the pairs
        while True:
            top_count, top_bpe_pairs = counter_index.peekitem()
            if len(top_bpe_pairs) > 0:
                break
            del counter_index[top_count]
        top_bpe_pairs = sorted(list(top_bpe_pairs))
        bpe_pair = top_bpe_pairs[-1]  # select the largest one to merge
        bpe_pairs_merged.append(bpe_pair)
        vocab[len(vocab)] = bpe_pair[0] + bpe_pair[1]
        # incrementally update the pairs that overlapped
        # I haven't come up with a very good solution here yet, but a relative brute force
        # solution is to:
        #   1. first find the original pretoken that is affected
        #   2. use the original pretoken to find the merged pretoken
        #   3. loop through the pretoken and identify the bpe pairs that overlap
        #   4. update the counter
        tokens = list(pair_to_token[bpe_pair])  # tokens need to be merged
        for token in tokens:  # each token is a tuple of bytes
            current_token = original_to_merged[token]
            # need to find all starting index of the merged pair
            indices = []
            for i in range(len(current_token) - 1):
                if current_token[i] == bpe_pair[0] and current_token[i+1] == bpe_pair[1]:
                    indices.append(i)

            filtered_indices = []
            for i in indices:
                if not filtered_indices or i - 1 > filtered_indices[-1]:
                    filtered_indices.append(i)

            merged_token = merge_token(current_token, bpe_pair, filtered_indices)
            original_to_merged[token] = merged_token

            update_bpe_pairs(
                current_token, 
                merged_token, 
                token,
                pretoken_counter[token], 
                bpe_pairs, 
                counter_index, 
                pair_to_token,
            )
        # clean up the merged bpe pair
        del pair_to_token[bpe_pair]
    
    return vocab, bpe_pairs_merged

if __name__ == "__main__":
    import pickle

    input_path = "/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # input_path = "/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/data/owt_train.txt"
    
    vocab, bpe_pairs_merged = train_bpe(input_path, 10000, ["<|endoftext|>"])

    with open("/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/result/vocab-owt.pkl", "wb+") as file:
        pickle.dump(vocab, file)
    
    with open("/Users/yangpei/Desktop/side-projects/cs336/cs336-assignment1-basics/result/merges.pkl", "wb+") as file:
        pickle.dump(bpe_pairs_merged, file)

    
