import os
import regex as re
import heapq

from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # dict[int, bytes] => the tokenizer vacobulary, a mapping from int (token id) to bytes
    # list[tuple[bytes, bytes]] => a list of BPE merges, each list (<token1>, <token2>) means token1 should be merged with token2
    tokenizer_vocabs = dict()
    merges = list()
    num_processes = kwargs.get("num_processes", os.cpu_count())
#     num_processes = 1
    
    # Initialize the vocabularies 
    # Append special tokens 
    for token in special_tokens:
        tokenizer_vocabs[len(tokenizer_vocabs)] = token.encode("utf-8")
    # Append the 256 characters which can be represented by one byte
    for i in range(256):
        tokenizer_vocabs[len(tokenizer_vocabs)] = bytes([i])
  
    # Pre-tokenizations
    pre_tokens = dict()
    with open(input_path, "rb") as f:
        split_special_token = special_tokens[0] if special_tokens else "<|endoftext|>"
        boundaries = find_chunk_boundaries(
            f, num_processes, split_special_token.encode("utf-8"))

    # Prepare list of (input_path, start, end) arguments
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    # Use multiprocessing.Pool
    with Pool(processes=num_processes) as pool:
        chunk_counters = pool.starmap(process_chunk_for_bpe, chunk_args)
        
    pre_tokens_counter = sum(chunk_counters, Counter())
    pre_tokens = {k.encode("utf-8") : v for k, v in pre_tokens_counter.items()}
        
    # Compute the merges
    # Initialize token sequences
    pre_token_sequences = defaultdict(list)
    for pre_token, _ in pre_tokens.items():
        for i in range(len(pre_token)):
            pre_token_sequences[pre_token].append(bytes([pre_token[i]]))
            
    while len(tokenizer_vocabs.keys()) < vocab_size:
        # Count adjacent pairs
        pair_freq = Counter()
        for pre_token, seq in pre_token_sequences.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freq[pair] += pre_tokens[pre_token]
        # Find the pair with highest frequency to create new token sequences
        if len(pair_freq) == 0:
            break
            
        max_count = max(pair_freq.values())
        most_frequent = max([k for k, v in pair_freq.items() if v == max_count])

        # Add the new token in the tokenizer vocabularies 
        tokenizer_vocabs[len(tokenizer_vocabs)] = most_frequent[0] + most_frequent[1]
        # Merge the new pair to update the token sequences
        new_pre_token_sequences = defaultdict(list)
        for pre_token, seq in pre_token_sequences.items():
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == most_frequent:
                    new_pre_token_sequences[pre_token].append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    new_pre_token_sequences[pre_token].append(seq[i])
                    i += 1
        pre_token_sequences = new_pre_token_sequences
        merges.append(most_frequent)

    return tokenizer_vocabs, merges


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

def process_chunk_for_bpe(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # escape special tokens first since `|` means differently in regex
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        # strip out special tokens
        parts = re.split("|".join(escaped_tokens), chunk)
        result = Counter()
        # NOTE: We should not do pre-tokenization across parts split by special tokens
        for p in parts:
            result += pre_tokenization(p)
        return result


def pre_tokenization(chunk) -> Counter:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counter = Counter()
    
    for match in re.finditer(PAT, chunk):
        word = match.group()
        counter[word] += 1
    
    return counter
