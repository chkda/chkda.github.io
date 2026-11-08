---
title: "PagedAttention, Part 1: Block Allocation"
date: 2026-11-08
# weight: 1
# aliases: ["/first"]
tags: ["llm","quantization"]
author: "Chhaya Kumar Das"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
# description: "Desc Text."
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: true
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: false
UseHugoToc: false
# cover:
#     image: "<image path/url>" # image path/url
#     alt: "<alt text>" # alt text
#     caption: "<text>" # display caption under cover
#     relative: false # when using page bundles set this to true
#     hidden: true # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link

---

In this first part, we will look at why we need a KV cache and how to allocate blocks for it. We will use a small implementation to follow how a request gets its blocks.

# Why do we need a KV cache?

Before we get to PagedAttention, let us look at how an autoregressive language model generates text. It predicts one token at a time, adds that token to the input, and then predicts the next token.

Suppose we give the model the prompt `The cat`. To keep the example simple, assume each word is one token:

```text
The cat          → sat
The cat sat      → on
The cat sat on   → the
```

The input grows by one token at each step. Without caching, we would pass the whole input through the model again every time. This means repeating calculations for `The`, `cat`, and the other tokens we have already processed.

But do we need to repeat that work?

## What can we reuse?

In a causal Transformer, each token can only use information from itself and the tokens before it. So adding `sat` after `The cat` does not change the calculations we already did for `The` and `cat`. We can reuse that work.

Each attention layer calculates three vectors for each token: a **query (Q)**, a **key (K)**, and a **value (V)**. The new token's query is compared with the keys to decide how much attention to give each token. These attention weights are then used to combine the values.

To process a new token, we need its query, key, and value, plus the keys and values of the earlier tokens. **We do not need to calculate the earlier tokens' keys and values again.** We save them in memory and reuse them. This is called the **KV cache**, and each attention layer has its own cache.

Generation then has two stages:

1. **Prefill:** Process the prompt and save its keys and values in the cache. The output for the last prompt token gives us the prediction for the first new token.
2. **Decode:** Process the new token and add its keys and values to the cache at each layer. Use its output to predict the next token, then repeat.

## How much memory does the KV cache need?

For one request, the KV cache stores a key and a value for every cached token, at every attention layer. Its memory usage is:

```text
KV cache memory (bytes) = 2 × L × T × H_kv × D × B
```

Here:

- **2** accounts for both keys and values.
- **L** is the number of attention layers.
- **T** is the number of cached tokens, including the prompt and the generated tokens processed so far.
- **H_kv** is the number of key/value heads per layer.
- **D** is the size of each head.
- **B** is the number of bytes used to store each value, such as 2 bytes for FP16 or BF16.

For example, take a model with **32 layers**, **32 KV heads** per layer, and a **head size of 128**. If one request has **4,096 cached tokens** stored in **FP16**, it needs:

```text
2 × 32 × 4,096 × 32 × 128 × 2
= 2,147,483,648 bytes
= 2 GiB
```

That is **2 GiB just for one request's keys and values**, before any extra memory used to manage the cache. Models with fewer KV heads use less cache memory; use the KV head count in this formula, which can differ from the query head count.

With many requests running in parallel, these memory needs add up. We know each prompt's length when it arrives, but we do not know exactly how many tokens the model will generate. Reserving space for each request's maximum possible length can waste memory, while reserving too little means we need to find more space as it grows.

# PagedAttention

PagedAttention lets us split the KV cache into fixed-size blocks and allocate them as a request grows.

![Animation showing how logical KV cache blocks map to physical blocks in GPU memory.](/images/paged_attention.gif)

## Logical blocks and physical blocks

- **Logical blocks** group a request's tokens in sequence order. With a block size of 4 tokens, logical block 0 holds the KV entries for tokens 0–3, block 1 holds tokens 4–7, and so on.
- **Physical blocks** are the actual spaces in GPU memory that store these keys and values. A request's physical blocks do not need to sit next to each other.

A **block table** maps each logical block to its physical block. For example, logical blocks `0, 1, 2` might map to physical blocks `7, 2, 10`. Attention follows this table to find the right keys and values.

When the current block fills up, we allocate another free physical block and update the table. When the request finishes, its blocks can be reused. This lets requests grow without reserving one large, continuous chunk of memory upfront.

# A small implementation: LiteCache

Let us make this concrete with [LiteCache](https://github.com/chkda/litecache), a small implementation I built to explore these ideas. For this part, we will focus on **setting up the memory pool, allocating physical blocks, and updating a request's block table as it grows**. The snippets below are shortened to focus on these steps.

## Setting up the memory pool

[`PagedAttentionCache`](https://github.com/chkda/litecache/blob/main/litecache/cache/paged_attention.py) starts by creating separate key and value tensors for each attention layer. Each tensor has this shape:

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

Here, `num_blocks` is the total number of physical blocks in the pool, and `block_size` is the number of tokens each block can hold.

**The whole pool is allocated upfront.** As requests arrive and grow, we assign blocks from this pool to them. Allocating a block to a request means choosing an available block ID; the tensor memory already exists.

## Picking a free block

The [`BlockAllocator`](https://github.com/chkda/litecache/blob/main/litecache/memory/allocator.py) tracks two sets: free block IDs and allocated block IDs. At the start, every block ID is free. Its `allocate()` method picks one:

```python
def allocate(self) -> int:
    if len(self.__free_blocks) == 0:
        raise ValueError("No free blocks")
    block_id = self.__free_blocks.pop()
    self.__allocated_blocks.add(block_id)
    return block_id
```

The returned ID tells us which part of the pool a request can use. Since the free IDs are stored in a set, we do not depend on getting them in any particular order.

## Giving a request its blocks

The [`SequenceManager`](https://github.com/chkda/litecache/blob/main/litecache/memory/sequence.py) tracks each request using three fields:

```python
class Sequence:
    seq_id: int
    num_tokens: int
    block_table: List[int]
```

Suppose a request starts with **10 tokens**, and each block holds **4 tokens**. We need to round up when calculating the number of blocks:

```python
num_blocks = (num_tokens + block_size - 1) // block_size
# (10 + 4 - 1) // 4 = 3 blocks
```

The manager checks that enough blocks are free, then asks the allocator for three physical block IDs:

```python
if not self.allocator.can_allocate(num_blocks):
    raise RuntimeError("Cannot allocate sequence. Not enough free blocks")

block_table = []
for _ in range(num_blocks):
    block_id = self.allocator.allocate()
    block_table.append(block_id)
```

Checking first avoids taking some blocks and then discovering that there are not enough for the request. The manager saves the resulting block table along with the request's ID and token count.

Suppose it receives `[7, 2, 10]`:

| Logical block | Token positions | Physical block |
| --- | --- | --- |
| 0 | 0–3 | 7 |
| 1 | 4–7 | 2 |
| 2 | 8–11 | 10 |

Positions 8 and 9 have slots reserved in the last block. Its remaining two slots are available for future tokens. These blocks are now assigned to this request, so the allocator will not give them to another request while they are in use.

We now have space for the request's KV entries and a table that tells us where each logical block belongs. Allocating these blocks has not written any keys or values yet.

## What happens when new tokens are generated?

As each generated token is processed, we need a slot for its keys and values. If the last block still has room, we use its next empty slot. The block table stays the same.

Let us continue with our **10-token request**, a **block size of 4**, and the table `[7, 2, 10]`:

| Token count after adding a token | New token's position | Physical block | Slot within the block | Block table |
| --- | --- | --- | --- | --- |
| 11 | 10 | 10 | 2 | `[7, 2, 10]` |
| 12 | 11 | 10 | 3 | `[7, 2, 10]` |
| 13 | 12 | 5 (newly allocated) | 0 | `[7, 2, 10, 5]` |

Token positions and slots start at 0. At 12 tokens, all three blocks are full. When we make room for the 13th token, we ask the allocator for another free block. Suppose it returns **block 5**. We append that ID to the table, so **logical block 3 now maps to physical block 5**.

**The existing block IDs do not change.** We only add a new mapping at the end; the earlier keys and values stay where they are.

In LiteCache, `append_slots()` passes this work to `SequenceManager.append_tokens()`. The main steps are:

```python
old_num_blocks = len(seq.block_table)
seq.num_tokens += num_new_tokens
num_blocks = (seq.num_tokens + self.block_size - 1) // self.block_size
num_new_blocks = num_blocks - old_num_blocks

if num_new_blocks == 0:
    return

if not self.allocator.can_allocate(num_new_blocks):
    raise RuntimeError("Cannot allocate sequence. Not enough free blocks")

for _ in range(num_new_blocks):
    block_id = self.allocator.allocate()
    seq.block_table.append(block_id)
```

This lets us reserve space one block at a time as the request grows, instead of reserving space for its maximum possible length at the start. The pool still has a fixed capacity: if no blocks are free, this implementation raises an error.

In the next part, we will use this table to write a token's KV entries and read them for attention. We will also look at returning blocks when a request finishes.

# References

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm)
- [LiteCache source code](https://github.com/chkda/litecache)
