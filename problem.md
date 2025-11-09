# Written Problem Answers

## Understanding Unicode

- (a) `\x00`
- (b) The print is just nothing, while `__repr__()` returns `"'\\x00'"`
- (c) In the string it would be converted as `\x00` while when print it would be nothing

## Unicode Encodings

- (a) The encoded bytes of "uft-8" is more *clean* compared to "utf-16" and "utf-32" (need to to some more research on this question)
- (b) The problem lays in that the function is decoding each encoded bytes individually, where this is not correct as some characters might use multiple bytes to represent. One example could be *这个很牛逼*
- (c) `b'\x00\x80'`, use a python scripts to enumerate all possible combination and find one, see `scripts.py`

## BPE training - TinyStory

- (a): Using 8 processes for pretokenization and the overall training finished within 1 minutes; the each process takes around 1.3GB memory; the longest token is `b' responsibility'`
- (b): most of time is spend on the pretokenization step

```bash
Program: cs336_basics/train_bpe.py

56.386 <module>  train_bpe.py:1
└─ 56.340 train_bpe  train_bpe.py:160
   ├─ 48.359 Process.join  multiprocessing/process.py:142
   │     [2 frames hidden]  multiprocessing
   │        48.359 waitpid  <built-in>
   └─ 7.125 update_bpe_pairs  train_bpe.py:138
      ├─ 5.658 update_counter  train_bpe.py:103
      │  ├─ 1.985 SortedDict.__setitem__  sortedcontainers/sorteddict.py:280
      │  │  └─ 1.592 SortedList.add  sortedcontainers/sortedlist.py:253
      │  ├─ 1.917 SortedDict.__delitem__  sortedcontainers/sorteddict.py:232
      │  │  └─ 1.478 SortedList.remove  sortedcontainers/sortedlist.py:426
      │  └─ 1.366 [self]  train_bpe.py
      └─ 1.076 [self]  train_bpe.py
```

## BPE training - OWT

- (a): total running time is around 18 minutes, total number of pretoken 6601892, peak memory usage is around 7GB, the longest token is `b'----------------------------------------------------------------'`
- (b): TODO

## BPE tokenizer

TODO