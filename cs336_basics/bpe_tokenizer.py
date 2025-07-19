
import json
import ast
import regex as re
from typing import Iterator, Iterable
from cs336_basics.train_bpe import train_bpe

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.special_tokens = special_tokens if special_tokens else list()
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.vocab = vocab
        # Add special tokens into the vocabulary
        # for t in self.special_tokens:
        #    self.vocab[len(self.vocab)] = t.encode("utf-8")
        self.bytes_to_index = {v : k for k, v in self.vocab.items()}
        # Store the merges as rank map to quickly figure out what's the rank for one merge
        self.merges_rank = {merges[i] : i for i in range(len(merges))}
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        assert vocab_filepath == merges_filepath, "The current implementation assume the vocab and merges are stored in the same path"
        with open(vocab_filepath, 'r') as f:
            data = json.load(f)
        
        vocab = {int(k): ast.literal_eval(v) for k, v in data['vocab'].items()}
        merges = [tuple(ast.literal_eval(x) for x in pair) for pair in data['merges']]
        
        return cls(vocab, merges, special_tokens)

    def encode_ordinary_text(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, text)
        
        result = list()
        for pre_token in pre_tokens:
            # NOTE: Encoding the string first then transfer to the list of bytes,
            # other once one item in the seq may include multiple bytes
            seq = [bytes([b]) for b in pre_token.encode("utf-8")]
            # merge the pretokens based on the trained merges
            while True:
                # Find the best rank of pair to merge 
                best_rank, best_pos = float('inf'), None
                for i in range(len(seq) - 1):
                    rank = self.merges_rank.get((seq[i], seq[i + 1]), float('inf'))
                    if rank < best_rank:
                        best_rank, best_pos = rank, i
                
                if best_pos is None:
                    break
                else:
                    seq = seq[:best_pos] + [seq[best_pos] + seq[best_pos + 1]] + seq[best_pos + 2:]
                    
            for b in seq:
                result.append(self.bytes_to_index[b])
        return result 
    
    def encode(self, text: str) -> list[int]:
        escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
        delimiter = "|".join(escaped_tokens)
        parts = re.split(f"({delimiter})", text)
        
        result = list()
        for p in parts:
            if p in self.special_tokens:
                result.append(self.bytes_to_index[p.encode("utf-8")])
            else:
                result.extend(self.encode_ordinary_text(p))
                
        return result
        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        result = list()
        
        for text in iterable:
            result += self.encode(text)
        
        return result
    
    def decode(self, ids: list[int]) -> str:
        b = b""
        for i in ids:
            b += self.vocab[i]
        return b.decode("utf-8", errors="replace")
