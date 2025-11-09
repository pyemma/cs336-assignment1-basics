from itertools import chain
import os
from typing import BinaryIO, Iterable
import regex as re

from cs336_basics.constants import PAT


class Tokenizer:

    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # sort the special tokens by length to match to the longest first
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }

        # add special tokens into vocab
        for special_token in self.special_tokens:
            if special_token.encode("utf-8") not in self.reverse_vocab:
                self.vocab[len(self.vocab)] = special_token.encode("utf-8")

        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }

        # index the merges
        self.merges_index = {
            merge: idx for idx, merge in enumerate(merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def merge_bpe(self, bpe: list[bytes]) -> list[bytes]:
        """
            Merge the bpe based on the order of merges
        """
        merged_bpe = bpe.copy()  # side effect free merge
        while True: # loop until there is no more merges
            indices = []  # the starting index of the identified bpe pairs
            for i in range(len(merged_bpe) - 1):
                if (merged_bpe[i], merged_bpe[i+1]) in self.merges_index:
                    indices.append((self.merges_index[(merged_bpe[i], merged_bpe[i+1])], i))
            
            if not indices:  # no more merges identified
                break
            
            indices = sorted(indices) # sort the indices to get the smallest merge
            bpe_pair, index = indices[0]
            new_merged_bpe = merged_bpe[:index] + [merged_bpe[index] + merged_bpe[index+1]] + merged_bpe[index+2:]
            merged_bpe = new_merged_bpe
        return merged_bpe

    def encode(self, text: str) -> list[int]:
        """
            1. pretokenize the text
            2. iteratively find the merges based on the order of merges
            3. until no more merges could be find
            4. convert to ids
        """
        if not text:
            return []
        if self.special_tokens:
            # first split based on special tokens to avoid incorrectly merges
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            # add `()` to also return the pattern matching group for easier handle of special tokens
            docs = re.split("(" + "|".join(escaped_special_tokens) + ")", text)
        else:
            docs = [text]

        token_ids = []
        for doc in docs:
            # special token could also be in the splits
            if doc in self.special_tokens:
                token_ids.append([self.reverse_vocab[doc.encode('utf-8')]])
                continue
            matches_iterator = re.finditer(PAT, doc)
            ids = []
            for match in matches_iterator:
                s = match.group()
                bpe = [bytes([b]) for b in s.encode("utf-8")]
                merged_bpe = self.merge_bpe(bpe)
                # after merge, convert to ids
                ids = [self.reverse_vocab[p] for p in merged_bpe]
                token_ids.append(ids)

        result = list(chain.from_iterable(token_ids))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
            The core idea is to chunk the words from the iterable and then encode
            when receive sufficient number of words
        """
        for i in iterable:
            result = self.encode(i)
            for id in result:
                yield id

    def decode(self, ids: list[int]) -> str:
        bpe = bytes([])
        for id in ids:
            bpe += self.vocab[id]
        return bpe.decode("utf-8", errors="replace")