import os
import regex as re
import logging
from tqdm import tqdm
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    
    # Initialize the vocabularies 
    # Append special tokens 
    for token in special_tokens:
        tokenizer_vocabs[len(tokenizer_vocabs)] = token.encode("utf-8")
    # Append the 256 characters which can be represented by one byte
    for i in range(256):
        tokenizer_vocabs[len(tokenizer_vocabs)] = bytes([i])
  
    # Pre-tokenizations
    pre_tokens = dict()
    logging.info("Chunking...")
    with open(input_path, "rb") as f:
        split_special_token = special_tokens[0] if special_tokens else "<|endoftext|>"
        boundaries = find_chunk_boundaries(
            f, num_processes, split_special_token.encode("utf-8"))

    # Prepare list of (input_path, start, end) arguments
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    # Use multiprocessing.Pool
    logging.info("Pre-tokenization...")
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
    
    # Initialize the pair frequences
    pair_freq = Counter()
    token_ocurrences = defaultdict(set)
    for pre_token, seq in pre_token_sequences.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_freq[pair] += pre_tokens[pre_token]
            token_ocurrences[seq[i]].add(pre_token)
            token_ocurrences[seq[i + 1]].add(pre_token)
            
    total_iterations = vocab_size - len(tokenizer_vocabs.keys())
    pbar = tqdm(total=total_iterations, desc="Merging", ncols=70)
    while len(tokenizer_vocabs.keys()) < vocab_size:
        if len(pair_freq) == 0:
            break
        # Find the pair with highest frequency to create new token sequences
        max_count = max(pair_freq.values())
        most_frequent = max([k for k, v in pair_freq.items() if v == max_count])
        
        # Append the new merge
        merges.append(most_frequent)
        
        # Add the new token in the tokenizer vocabularies 
        new_token = most_frequent[0] + most_frequent[1]
        tokenizer_vocabs[len(tokenizer_vocabs)] = new_token
        
        # Set the frequencies of the merged pair tokens to 0
        pair_freq[most_frequent] = 0
        
        # Update the token sequences of the impacted pre tokens 
        impacted_pre_tokens = token_ocurrences[most_frequent[0]] | token_ocurrences[most_frequent[1]]
        for pre_token in impacted_pre_tokens:
            seq = pre_token_sequences[pre_token]
            # Rebuild the pre-token sequences for the impacted pre-token
            i = 0
            new_seq = list()
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == most_frequent:
                    new_seq.append(new_token)
                    token_ocurrences[new_token].add(pre_token)
                    if i - 1 >= 0:
                        pair_freq[(seq[i - 1], seq[i])] -= pre_tokens[pre_token]
                        pair_freq[(seq[i - 1], new_token)] += pre_tokens[pre_token]
                    if i + 2 < len(seq):
                        pair_freq[(seq[i + 1], seq[i + 2])] -= pre_tokens[pre_token]
                        pair_freq[(new_token, seq[i + 2])] += pre_tokens[pre_token]
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            pre_token_sequences[pre_token] = new_seq
        
        pbar.update(1)
    pbar.close()

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
