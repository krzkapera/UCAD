# Everything else we measured

`FINDINGS.md` argues one thing and `REPRODUCTION.md` reproduces the published tables. This file is the
rest: measurements that belong to neither, kept because several of them took real compute and because
two of them contradict claims we made earlier and later withdrew.

Numbers are image AUROC / pixel AUPR. Unless stated otherwise: reference implementation, ViT-B/16
ImageNet-21k weights, block 5, bank 196, 224px, one seed, VisA on the official 1cls split.

**Read the labels on the VisA rows.** The grid, bank-size and block sweeps below predate the discovery
that the official split is the right one, so they were run on the per-category folder copy with the
`augreg_in21k_ft_in1k` checkpoint. They are internally consistent and comparable to each other, and
they are *not* comparable to the tables in `REPRODUCTION.md`, which are on the official split with the
checkpoint the paper states. Each affected table says so.

## Which pretrained checkpoint

The paper says "pretrained on ImageNet 21K" and `default_cfgs` points at
`imagenet21k/ViT-B_16.npz`, which agrees. Three readings of "pretrained" are available in timm under
the same model name, and they are not interchangeable.

VisA, 25 epochs, three seeds, on the **per-category folder copy** (not the official split):

| checkpoint | VisA |
|---|---|
| `orig_in21k` - the active line, and what the paper states | 0.8069 +- 0.0126 |
| `augreg_in21k_ft_in1k` - the line commented out beside it | 0.8634 +- 0.0150 |
| `augreg2_in21k_ft_in1k` - what a current timm resolves the name to | 0.8725 +- 0.0046 |
| paper | 0.874 |

Read on averages alone this looks like the paper matches the ImageNet-1k fine-tunes, which the paper
does not claim. Read per category it says the opposite. Mean absolute deviation from the published
per-category tables:

| | VisA | MVTec |
|---|---|---|
| `orig_in21k` | 0.0801 | **0.0089** |
| `augreg_in21k_ft_in1k` | 0.0366 | 0.0338 |
| `augreg2_in21k_ft_in1k` | 0.0346 | 0.0307 |

MVTec identifies the checkpoint unambiguously - `orig_in21k` matches ten of fifteen categories within
0.01 - while VisA on the folder copy matches none of them. That was the clue that the VisA gap was the
data split rather than the weights, and switching to the official split brought VisA's deviation from
0.080 to 0.017 with the same weights. The lesson worth keeping: **an average can agree while every
category disagrees, so check per category before concluding that a configuration is right.**

## Feature grid resolution

The grid was hardcoded at 14x14 in the loss and in two reshapes, so this needed those to be derived
from the model first. MVTec, five categories (screw, cable, hazelnut, grid, transistor), untrained,
`augreg_in21k_ft_in1k`. Every VisA number in this section and the next is on the **folder copy**:

| grid | bank 196 | bank = tokens per image |
|---|---|---|
| 7x7 (patch32) | 0.7659 | - |
| 14x14 (patch16) | 0.8272 | 0.8481 (784) |
| 24x24 (patch16 at 384px) | 0.8154 | 0.8367 (576) |
| 28x28 (patch8) | 0.7575 | **0.8633** (784) |

At a fixed 196 vectors a finer grid **loses**, because the coreset then compresses four times harder;
give the bank room and the ordering reverses. This is why nobody would find the effect while
`memory_size` stays at its published value.

At equal bank size the finer grid helps MVTec and hurts VisA. Full benchmarks, untrained, bank 784:

| block | MVTec 14x14 | MVTec 28x28 | VisA (folder copy) 14x14 | VisA (folder copy) 28x28 |
|---|---|---|---|---|
| 5 | 0.9366 | 0.9427 | 0.8516 | 0.8504 |
| 7 | 0.9577 | **0.9639** | **0.8535** | 0.8321 |
| 9 | 0.9543 | 0.9535 | 0.8252 | 0.8235 |

That fits MVTec's defects being smaller than one patch of the coarse grid and VisA's being larger and
more contextual. Comparing 24x24 against 28x28 separates a finer patch from simply more tokens: they
have similar token counts and behave differently, so it is the patch scale that matters, not the
sequence length.

## Bank size

The single largest lever in the method, and the paper does sweep it (Table 6) up to 4x. Untrained,
block 7:

| bank | VisA (folder copy), 14x14 | MVTec, 28x28 |
|---|---|---|
| 196 | 0.8122 | - |
| 392 | 0.8198 | 0.9524 |
| 784 | 0.8535 | 0.9639 |
| 1568 | 0.8673 | 0.9690 |
| 1960 | 0.8802 | - |
| 3136 | 0.8826 | 0.9716 |

Still rising at 3136. Two consequences worth keeping:

**Untrained configurations beat the published numbers**, at more memory: MVTec 0.9639 / 0.5312 at
block 7, 28x28, bank 784 against 0.930 / 0.456 published, and on the folder copy VisA 0.8826 at
block 7, bank 3136 against 0.874. At the paper's own budget of 196 the untrained model still beats the
published MVTec figure (0.9395 at block 9) but not the VisA one (0.8122 at block 7, folder copy).

On the official split with the checkpoint the paper states, the same point holds at 196: untrained
0.7872 against 0.874 published, so VisA needs the extra memory to overtake it, while MVTec does not.

**The 1960-vector reading is computed in every run and never reported.** `--basic_size` defaults to
1960 and its results go to `results_nolimit/`. On VisA an untrained model with that bank scores 0.8867
- above the 0.874 the paper reports for the trained one, on the same split, in a file the code has
always written.

None of this is free: 784 is four times the memory the method advertises and 1960 is ten times, and
fixed small memory is the method's selling point. The honest way to read it is that the published
figures measure a memory budget at least as much as they measure a loss.

## Where the paper and the code disagree, beyond the loss

**The prompt mechanism.** The paper adds a prompt to each layer's input, `k^i = f^i(k^{i-1} + p^i)`,
and accounts for it as (15, 7, 768) floats. The stated total of 23.28MB confirms the shape - key
(15, 196, 1024) plus knowledge (15, 196, 1024) plus prompt (15, 7, 768), times four bytes, is 23.28
MiB exactly. But (15, 7, 768) admits two readings: 7 tokens per task, or one 768-vector added at each
of 7 layers. The additive equation favours the second. The code does neither: prefix tuning into the
attention keys and values of twelve layers, 24x768 per task, which is 3.4x more than the paper's own
memory accounting allows for.

Prompt length is the one part of this we could test. Over 25 epochs on five MVTec categories with the
`augreg_in21k_ft_in1k` checkpoint: length 1 gives 0.8396, length 5 gives 0.8562, length 7 gives
0.8463. The direction matches the paper, the
magnitude explains nothing.

**Re-weighting.** The paper's Eq. 5-6 re-weight the nearest-neighbour distance by its neighbours. The
CLI default `--anomaly_scorer_num_nn 5` would enable that; the README command passes 1, which disables
it. **We never tested 5.**

**Patch neighbourhood.** `--patchsize` defaults to 3 and the README passes 1. **We never tested 3.**

## The SAM label map

`cv2.resize` without an interpolation argument averages the segment ids the map holds, and the loss
compares those ids with `==`, so cells on segment boundaries end up matching nothing. Sampling them
instead (`UCAD_SAM_INTERP=nearest`) is worth +0.008 image AUROC and -0.006 pixel AUPR over 25 epochs
on five MVTec categories with the `augreg_in21k_ft_in1k` checkpoint, and -0.002 at a 28x28 grid, where
we had expected more of it because a finer grid has more segment boundaries. Real, small, and not an
explanation for anything.

We also ran both benchmarks against a second mask source (SAM2 instead of SAM-B) in the pyCLAD
implementation earlier: no systematic effect on whether training helps.

## Claims we made and withdrew

Recorded because the reasoning that produced them is worth not repeating.

**"The bank size was never explored."** False - Table 6 is exactly that ablation. The correct claim is
narrower: the 1960-vector reading is computed and not reported.

**"Nothing in the paper argues for block 5."** False - Table 7 sweeps blocks 1, 3, 5, 7, 9 and the
paper says it kept 5 "for simplicity". Our own reproduction of that table shows the best block beats 5
by 0.006 on MVTec and 0.010 on VisA, not the +0.022 an earlier five-category run suggested.

**"The published number matches the checkpoint the code disables."** Withdrawn: that rested on VisA
averages measured on the wrong split. The paper states ImageNet-21k, the code configures ImageNet-21k,
and per category MVTec confirms it.

**"CPM is a large real contribution."** Withdrawn as stated. Its +0.20 is measured against a baseline
that discards earlier concepts; against one shared bank at the same total memory it adds nothing. See
`FINDINGS.md`.

**"screw is out of reach for both implementations."** Withdrawn: that was measured with the
ImageNet-1k fine-tuned weights, which give 0.61-0.62. With the checkpoint the paper states, screw
reaches 0.7039 against 0.739 published.

## Open questions

Things we can state but not explain:

- Where the published Forgetting Measure of 0.010 (MVTec) and 0.039 (VisA) comes from. This
  architecture shares nothing between concepts and routes every test image correctly, so the matrix
  Eq. 7 is defined over is constant along its rows and the measure is exactly 0.
- Why block 1 misses by +0.080 on MVTec in Table 7 when blocks 3 to 9 land within 0.005.
- Why the MVTec no-SCL cell at bank 196 misses by +0.021 when the 392 and 784 cells land within 0.002.
- Why pcb4 (-0.066), pcb3 (-0.042) and screw (-0.035) deviate when the other categories agree to 0.01.
- Whether the authors ran a version with the inference phase uncommented.
- What the `train()` method and `get_normal_prototypes_seg` are for. They build k-means prototypes
  over groups of four images, are not on the path `run_ucad.py` takes, would break for a batch size
  that is not a multiple of four, and the paper does not describe them. `prototype_size=5` is passed
  to the model with the comment `# failure version`.
