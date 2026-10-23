---
title: "LoRA and QLoRA: Elixir for the GPU poor"
date: 2025-01-15
# weight: 1
# aliases: ["/first"]
tags: ["llm","quantization"]
author: "Chhaya Kumar Das"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
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


# Why do we need LoRA and QLoRA?

Say I have a pretrained language model and want to adapt it to my own dataset. Maybe I want it to answer questions in a particular domain or follow a specific response format. I can fine-tune it, but how much GPU memory will that take?

With **full fine-tuning**, we update all the model's weights. Those weights are only part of the memory bill: training also needs gradients, optimizer states, and intermediate activations. A model that fits on my GPU for inference may still be too large to fine-tune on it.

For a concrete example, a 7-billion-parameter model stored at 16 bits per parameter needs roughly **14 GB just for its weights** (using decimal GB). That is before we account for the rest of training. Access to model weights alone does not make adapting them affordable.

This raises two questions. Can we train a much smaller set of parameters? And can we reduce the memory needed to hold the pretrained model itself?

These questions bring us to **LoRA (Low-Rank Adaptation)** and **QLoRA**. LoRA keeps the pretrained weights frozen and learns small trainable additions. QLoRA takes this further by combining LoRA with a frozen base model stored in 4-bit precision. We will start with LoRA, then build towards QLoRA. The original papers are linked at the end.

# LoRA: Low-Rank Adaptation

## What does LoRA set out to achieve?

LoRA's goal is to adapt a pretrained model while training far fewer parameters than full fine-tuning. Instead of learning a separate change for every entry in a large weight matrix, it represents the change using two smaller matrices.

Think of this as keeping the existing transformation and learning a small correction alongside it. The base weights stay frozen; the correction learns from our dataset. We can apply this to selected linear transformations in a Transformer, such as its attention projections.

## How does it work?

Let us zoom in on one linear transformation. Its pretrained weight matrix is **W₀**, and its input is **x**. Ignoring bias for simplicity, its original output is **W₀x**.

LoRA gives the same input a second path, through two trainable matrices called **A** and **B**. We add that path's output to the original output:

![LoRA equation: h = W₀x + (α/r)BAx. Shapes below each term: h is d × 1, W₀ is d × k, x is k × 1, B is d × r, and A is r × k. W₀ is frozen; B and A are trainable.](/images/lora-qlora/lora.svg)

*One adapted linear transformation. Only A and B are trained in this diagram; W₀ stays frozen.*

The forward-pass formula, with the original LoRA scaling, is:

```text
h = W₀x + (α/r) B(Ax)
```

Equivalently, the effective weight matrix is:

```text
W = W₀ + ΔW,     where ΔW = (α/r) BA
```

Here is what each symbol means:

| Symbol | Meaning | Shape |
| --- | --- | --- |
| x | Input vector | k |
| W₀ | Frozen pretrained weight matrix | d × k |
| A | Trainable matrix that maps the input to a smaller space | r × k |
| B | Trainable matrix that maps it back to the output space | d × r |
| r | Adapter rank, chosen much smaller than d and k | Scalar |
| α | Hyperparameter controlling the update's scale through α/r | Scalar |
| h | Output after adding both paths | d |

The order matters: **A acts first, then B**. Their product BA has the same shape as W₀, so the two transformations can be added. But BA has rank at most r. This is the “low-rank” part: we constrain the update, not the original pretrained matrix.

For example, take a **4096 × 4096** weight matrix. Updating all its entries means training **16,777,216 parameters**. With **r = 8**, A and B together contain only:

```text
(r × k) + (d × r)
= (8 × 4096) + (4096 × 8)
= 65,536 trainable parameters
```

That is **256 times fewer trainable parameters for this matrix**. It does not mean the whole training run uses 256 times less memory: the frozen model and activations still take space. This remaining cost is why quantizing the base model becomes interesting when we get to QLoRA.

## How does α/r control adapter strength?

In `h = W₀x + (α/r) B(Ax)`, the base output **W₀x** has a coefficient of 1, while **α/r** scales the adapter output. Think of it as a volume knob for the learned correction.

For a fixed A, B, and input x, with **r = 8**:

| α | α/r | Contribution added to W₀x |
| --- | --- | --- |
| 4 | 0.5 | Half of B(Ax) |
| 8 | 1 | B(Ax) unchanged |
| 16 | 2 | Twice B(Ax) |

A larger scale makes the same adapter update stronger; a smaller scale reduces its contribution. Setting the scale to zero removes the adapter's contribution entirely.

But **α/r = 2 does not mean the adapter is twice as strong as the base model**. The relative size of the two contributions also depends on W₀, A, B, and the input. This is an additive correction, not a percentage split between base and adapter.

During training, the scale also affects gradients flowing into A and B. Changing it can therefore change what the adapter learns. Likewise, increasing r changes the adapter's capacity as well as the denominator, so independently trained adapters at different ranks cannot be compared by α/r alone. The [original LoRA paper](https://arxiv.org/html/2106.09685v2#S4.SS1) uses this scaling to reduce the need to retune hyperparameters when varying r.

## Where is LoRA applied in a Transformer?

The expression above represents one linear transformation, not an entire Transformer block. A block contains several weight matrices we can choose to adapt:

| Part of the block | Possible LoRA targets | Example module names in Llama-style models |
| --- | --- | --- |
| Self-attention | Query, key, and value projections: Wq, Wk, Wv | `q_proj`, `k_proj`, `v_proj` |
| Attention output | Output projection: Wo | `o_proj` |
| Feed-forward network (MLP) | Up, down, and, in gated MLPs, gate projections | `up_proj`, `down_proj`, `gate_proj` |

A common starting point is to adapt the **query and value projections** across Transformer blocks. The [original LoRA paper](https://arxiv.org/abs/2106.09685) focused on attention weights and found adapting both Wq and Wv effective in its GPT-3 experiments. This is a choice of targets, not a restriction of LoRA.


Each targeted weight matrix gets its **own A and B pair**. The pretrained weights remain frozen. Targeting more matrices gives the model more trainable capacity, but also increases adapter parameters and training memory. LoRA is not normally attached to the softmax or activation functions themselves; here we are adapting the learned linear projections around them.

## Merging LoRA weights for inference

During training, we keep W₀ frozen and learn A and B separately. Once training is complete, we can fold their contribution into a single weight matrix:

```text
W_merged = W₀ + (α/r) BA
```

This works because BA and W₀ both have shape **d × k**. At inference time, the two paths become one:

```text
Before merging:  h = W₀x + (α/r) B(Ax)
After merging:   h = W_merged x
```

These expressions are mathematically equivalent for this linear adapter in evaluation mode, with dropout disabled; floating-point arithmetic can introduce small numerical differences. We do this for each adapted matrix. The merged matrix has the same shape as the original, so inference no longer needs the extra A and B operations. This is how [LoRA can avoid additional inference latency](https://arxiv.org/html/2106.09685v2#S4.SS1) after merging.

Merging is optional. Keeping adapters separate lets us store small task-specific checkpoints and swap them while sharing the same base model. A merged checkpoint instead contains the full adapted weights. Keep the original base and adapter files if you want to switch adapters or merge again later.

The base weights are frozen **during training**; merging afterwards deliberately creates updated inference weights. This simple equivalence assumes ordinary, unquantized linear weights. Merging into a quantized model involves additional precision considerations, which we will revisit with QLoRA.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
