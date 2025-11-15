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

## Transformer Accounting

- (a) We could have a breakdown of the parameters in each component, we are using single-precision floating point and thus it would take 4 bytes
  - Embedding: $50257 \times 1600 = 80411200$
  - For each transform block, we have 2 layer norm, 1 feed forward network and 1 causal self causal attention block
    - Self causal attention block
      - 4 linear projection, $4 \times 1600 \times 1600 = 10240000$ total parameters
      - 1 rope, $\times 1024 \times 1600  = 1638400$ non-learnable parameters
    - Layer norm
      - linear projection, $1600$ parameters
    - Feed froward network
      - 3 linear projection, $3 \times 1600 \times 6400 = 30720000$ total parameters
    - Total parameters in a transform block: $10240000 + 1600 \times 2 + 30720000 = 40963200$
  - For final layer norm, it is $1600$ parameters
  - For final projection head, it is the same as embedding $80411200$
  - Total parameters are $80411200 + 40963200 \times 48 + 1600 + 80411200 = 2127057600$
  - Store this in single precision floating points, gives us about 8GB total memory
- (b) If we only take a look at the matrix multiplication, then the embedding lookup part could be ignored
  - Self causal attention
    - 3 matrix multiplication, $3 \times 2 \times 1024 \times 1600 \times 1600 = 15728640000$ FLOPs
    - Query and key go through rope, contains 4 element-wise matrix multiplication and 2 element-wise matrix addition, $6 \times 1024 \times 800 = 4915200$, we could ignore this
    - Sdpa, query and key matrix multi on each head, $25 \times 2 \times 1024 \times 64 \times 1024 = 3355443200$ FLOPs; attention scores matrix multi with value, $25 \times 2 \times 1024 \times 1024 \times 64 = 3355443200$ FLOPs
    - Final output projection $2 \times 1024 \times 1600 \times 1600 = 5242880000$ FLOPs
    - Total self causal attention FLOPs $15728640000 + 3355443200 + 3355443200 + 5242880000 = 27682406400$
  - Layer norm only contains element-wise matrix multi, ignore
  - Feed forward network, 3 matrix multi, $3 \times 2 \times 1024 \times 1600 \times 6400 = 62914560000$ FLOPs
  - Total FLOPs in a transformer block $27682406400 + 62914560000 = 90596966400$
  - Final prediction head $2 \times 1024 \times 1600 \times 50257 = 164682137600$
  - Total FLOPs $90596966400 \times 48 + 164682137600 = 4513336524800$  
- (c) The feed forward part actually takes the majority of the FLOPs