---
title: "GPU Internals: A Speed Run"
date: 2026-07-05
# weight: 1
# aliases: ["/first"]
tags: ["GPU", "CUDA"]
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
This article is mostly focussed on CUDA devices. From here onwards whenever I mention GPU know it to be CUDA by NVIDIA.

All the notes that have been collected are for my self study. I will put sources below

## GPU Architecture

CUDA stands for Compute Unified Device Architecture.

![GPU/CPU architecture](/images/gpu-internal-speed-run/CPU-GPU-Arch.svg)

*Figure adapted from the [Modal GPU Glossary](https://modal.com/gpu-glossary/).*

GPU instructions are carried out by streaming multiprocessors (SMs).

An SM in NVIDIA is analogous to a CPU core. SMs execute computation and store state for computation in registers and caches.

## Streaming Multiprocessor Architecture

![H100_SM architecture](/images/gpu-internal-speed-run/SM_Architecture.svg)

*Figure adapted from the [Modal GPU Glossary](https://modal.com/gpu-glossary/).*

## GPU Core Concepts

- GPU cores are the primary compute units of SMs.
- GPU cores are Tensor Cores and CUDA Cores.
- A GPU core is not analogous to a CPU core.
- Consider GPU cores as pipes. Data goes in and transformed data comes out.
- SFU stands for Special Function Unit. It accelerates certain math functions such as `exp`, `sin`, and `cos`.
- LSU stands for Load/Store Unit. It dispatches requests to load and store data to GPU memory subsystems.
- LSUs interact with L1 data cache and global RAM indirectly.
- A warp is a group of threads that execute on the same clock cycle.
- The warp scheduler decides which group of threads executes on each clock cycle.
- CUDA Cores execute scalar arithmetic operations.
- Tensor Cores execute matrix operations.
- Groups of CUDA Cores are given the same instruction simultaneously, but the instruction is applied to different registers and data.
- Tensor Memory Accelerators (TMAs) are specialised hardware in Hopper and Blackwell to accelerate access to multidimensional arrays.
- A Texture Processing Cluster (TPC) is a pair of adjacent SMs.

## Memory Registers

- The register file is the primary store of bits in between manipulation by the cores.
- Analogous to CPU registers in terms of speed.
- It is split into 32-bit registers and can be allocated to different data types.
- L1 data cache is the private memory of an SM.
- It is accessed by LD/ST units of an SM.
- This cache is programmer-managed.
- Tensor Memory is specialised memory for storing inputs and outputs of Tensor Cores. It is only available in certain GPUs.
- GPU RAM is the global memory of a GPU. It can be either High Bandwidth Memory (HBM) or Double Data Rate (DDR) memory.


## Normal Data Flow and TMA Data Flow

### Normal data flow

```text
HBM / global memory
        ↓
       L2
        ↓
Load/store pipeline per thread register
        ↓
Load/store pipeline
        ↓
Shared memory
        ↓
Registers used for computation
        ↓
FP32 / Tensor Core
```

### TMA data flow

```text
HBM / global memory
        ↓
       L2
        ↓
      TMA
        ↓
Shared memory
        ↓
Registers used for computation
        ↓
FP32 / Tensor Core
```

Each thread has to calculate the memory address in global memory and then bring it to shared/L1 cache.

For data shared across threads, data has to be moved to shared memory.


## GPU Software Blocks

- Parallel Thread Execution (PTX) is an intermediate representation for code that runs on a CUDA device.
- Streaming Assembly (SASS) is the assembly format that runs on CUDA devices.
- Compute Capability is a versioning system that tells us what instructions can run on which CUDA device.
- A thread is the lowest unit of execution for a GPU.
- A single CUDA Core executes instructions from a single thread.
- A thread has its own registers.
- A warp is a group of threads that are scheduled together and execute in parallel.
- All threads in a warp are scheduled on the same SM.
- An SM normally executes multiple warps simultaneously.
- A warp group is a set of four contiguous warps.
- The entire set of four warps executes the same instruction. This is helpful for large matrix multiplications and is only present on Hopper architecture.

<!-- Diagram placeholder: Warp → warp group → four warps, each containing 32 threads -->

## Cooperative Thread Arrays

- A Cooperative Thread Array (CTA) is a collection of threads scheduled to the same SM.
- It is a common term for a thread block.
- A CTA can have many warp groups.
- Threads in the same CTA can coordinate with each other, but threads not in the same CTA coordinate through global memory.


## Grids, Blocks, and Threads

- A kernel is a typical unit of CUDA code, similar to functions in other languages for CPUs.
- A kernel is called or launched once and is executed many times, once by each thread.
- In CUDA C++, kernels take no parameters and return no value.
- A grid is a collection of blocks.
- A block is a collection of threads.

```text
Grid
└── Blocks
    └── Threads
```

- This is the typical kernel launch code `add<<1,4,256>>`We can have one grid with four blocks, and each block has 256 threads.

```text
Block 0 → SM 0
Block 1 → SM 1
Block 2 → SM 2
Block 3 → SM 3
```


```text
Block 0 and 4 → SM 0
Block 1 and 5 → SM 1
Block 2 and 6 → SM 0
Block 3 and 7 → SM 1
```

- Blocks have multiple warp groups.
- The blocks within a grid and threads within blocks can be arranged in one-, two-, or three-dimensional ways, such as `(1, 256)` or `(16, 16)`.
- Dimensions are for programming convenience.
- Blocks and grids are part of the programming model, while CTAs are a lower-level PTX representation.

## Memory Hierarchy

<!-- Diagram placeholder: Grid → VRAM → Block → L1 data cache → Thread → Register -->

- Registers hold information manipulated by a single thread and are normally stored in the register file of an SM.
- L1 data cache holds data and information for a limited block in an SM. This is also called shared memory.
- GPU VRAM is called global memory. It is accessible to all threads. All input during grid launch is placed here.


## References 

1. Modal, [GPU Glossary](https://modal.com/gpu-glossary/).