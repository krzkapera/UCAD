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

| | VisA (3 seeds) | MVTec |
|---|---|---|
| `orig_in21k` | 0.0801 | **0.0089** (3 seeds) |
| `augreg_in21k_ft_in1k` | 0.0366 | 0.0338 (1 seed) |
| `augreg2_in21k_ft_in1k` | 0.0346 | 0.0307 (3 seeds) |

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
published MVTec figure (0.9395 at block 9, on the `augreg_in21k_ft_in1k` checkpoint - we never ran
that combination with `orig_in21k`) but not the VisA one (0.8122 at block 7, folder copy).

On the official split with the checkpoint the paper states, the same point holds at 196: untrained
0.7872 against 0.874 published, so VisA needs the extra memory to overtake it, while MVTec does not.

**The 1960-vector reading is computed in every run and never reported as detection quality.**
`--basic_size` defaults to 1960 and its results go to `results_nolimit/`. On VisA an untrained model
with that bank scores 0.8867 - above the 0.874 the paper reports for the trained one, on the same
split, in a file the code has always written. It does reach the paper indirectly: its distance from
the 196-vector reading is what the README calls FM, and it matches the published Forgetting Measure.
See below.

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

**Routing is done per test set, not per image.** The paper's Eq. 4 assigns an individual test image to
a concept by comparing it against every stored key. The code instead scores a concept's whole test set
against each key bank and picks the argmin of the sum:

```python
                    query_scores, query_seg, labels_gt_query, masks_gt_query = PatchCore.predict(
                        dataloaders["testing"]
                    )
                    cur_query_list.append(np.sum(query_scores))
```

One decision per test set. That is a strictly easier problem than the per-image one - errors on
individual images cancel in the sum - and it needs the test images to arrive grouped by concept, which
is the assumption the method sets out to remove. Both versions turn out to be exact on these
benchmarks, but only the per-image version is what the paper claims, and it is the independent
implementation that measures it.

**Re-weighting is unreachable, because the flag that would enable it is dropped.** The paper's Eq. 5-6
re-weight the nearest-neighbour distance by its neighbours, which needs more than one neighbour.
`run_ucad.py` offers `--anomaly_scorer_num_nn`, defaulting to 5, and passes it on:

```python
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
```

but `PatchCore.load` spells the parameter without the "r":

```python
        anomaly_score_num_nn=1,
```

and its signature ends in `**kwargs`, so the misspelled argument is swallowed and the scorer is always
built with **one** neighbour whatever the flag says. Measured with the checkpoint the paper states,
`--anomaly_scorer_num_nn 5` against the `1` the README passes:

| | 5, untrained | 1, untrained | 5, 25 epochs | 1, 25 epochs |
|---|---|---|---|---|
| MVTec | 0.9153 / 0.4255 | 0.9153 / 0.4255 | 0.9254 / 0.4558 | 0.9254 / 0.4558 |
| VisA | 0.7872 / 0.2455 | 0.7872 / 0.2455 | 0.8668 / 0.2999 | 0.8668 / 0.2999 |

Bit for bit identical in all four columns, trained and untrained. The README's
`--anomaly_scorer_num_nn 1` is a no-op that happens to match the default, and Eq. 5-6 cannot be
switched on from the command line at all.

**Patch neighbourhood: the flag the README overrides is load-bearing.** `--patchsize` sets the
neighbourhood of patch features pooled into one descriptor. `PatchCore.load` defaults it to 3, PatchCore's
own value; the README passes 1. With the checkpoint the paper states:

| | patchsize 1 (README) | patchsize 3 (default) |
|---|---|---|
| MVTec, untrained | 0.9153 / 0.4255 | 0.6011 / 0.2196 |
| MVTec, 25 epochs | 0.9254 / 0.4558 | 0.6530 / 0.3407 |
| VisA, untrained | 0.7872 / 0.2455 | 0.5254 / 0.0454 |
| VisA, 25 epochs | 0.8668 / 0.2999 | 0.6144 / 0.0922 |

A 0.31 and 0.26 drop untrained, and 0.27 and 0.25 after 25 epochs under the full reporting protocol -
so neither training nor the epoch ensemble recovers it. Pooling a 3x3 neighbourhood of a 14x14 grid
mixes a third of the image into every descriptor, which is fatal for nearest-neighbour scoring. Omit
`--patchsize 1` and the method collapses to little better than chance on VisA - and neither the paper
nor the code comments say so.

**The published Forgetting Measure is a bank-size difference.** The README says so, in the paragraph
that explains why the inference phase is commented out:

> The inference involving a query process, it's slow, and I've commented it out in the code
> (./run_ucad.py lines 408-509). Training will directly provide the final results, and the inference
> process merely repeats this step. The final output will consist of two parts, with the lower
> metrics representing the final results, and the difference between them and the higher metrics
> results is denoted as FM.

The two parts are the two result directories every run writes: `results/` for the `--memory_size`
bank and `results_nolimit/` for the `--basic_size` one, which defaults to 1960. Measured in the
configuration this repository reproduces the paper with, 25 epochs of SCL:

| | bank 196 | bank 1960 | difference | published FM |
|---|---|---|---|---|
| MVTec | 0.9254 | 0.9354 | **0.0100** | **0.010** |
| VisA 1cls | 0.8668 | 0.9140 | **0.0472** | **0.039** |

MVTec is exact. VisA agrees at both ends: the paper implies a 1960-vector reading of 0.874 + 0.039 =
0.913 and we measure 0.9140, the same distance out as our 0.8668 against its 0.874.

So the quantity Table 3 reports as forgetting is the gain from a tenfold larger memory, measured on
one model against one test set. No earlier concept is re-scored anywhere in it, and Eq. 7 is not
evaluated. That resolves what was an open question here: the forgetting this architecture actually
exhibits, by the paper's own definition, is exactly 0, because nothing is shared between concepts and
routing is correct on every test image of both benchmarks.

Neither the paper nor the code says which of the two readings the tables carry. The `results/` one
matches, so the published detection figures are the 196-vector bank; the 1960-vector reading is
computed in every run and never reported as detection quality, only - if the README is taken at its
word - as the difference that becomes FM.

## Removing the prompt, and how it is initialised

Three ways of destroying the thing the paper is named after, all measured in the pyCLAD
implementation on the **per-category folder copy** of VisA with the `augreg2_in21k_ft_in1k`
checkpoint, single model, honest evaluation - so comparable to each other and not to the tables above.

**Initialisation makes no difference.** Untrained, one seed: the reference's `uniform(-1, 1)` gives
0.7984, all-zeros gives 0.7933. A 0.005 gap against a +-0.03 spread across seeds.

Worth recording how we know: the first all-zeros run came out byte-identical to the uniform one, which
is what told us the switch was a no-op rather than the finding. 0.7933 is the re-run.

**Deleting the prompt module costs nothing.** A zero-initialised prompt is not the same as no prompt:
prefix tuning prepends key and value tokens to the attention of all twelve blocks, so a zero prompt
still changes the softmax normalisation. Removing the tokens instead, three seeds:

| | image AUROC | pixel AP |
|---|---|---|
| no prompt module at all | 0.8099 +- 0.0133 | 0.2967 |
| prompt present, untrained | 0.7648 +- 0.0293 | 0.2781 |
| prompt present, 25 epochs of SCL | 0.7155 +- 0.0038 | 0.2052 |

Against the trained model the removal is worth +0.094 with disjoint ranges. Against the untrained one
the +0.045 is **not** significant - the ranges overlap, the two came from different driver scripts, and
the no-prompt variant draws no initialisation, which shifts the RNG stream and with it every coreset.
The claim the data supports is that removing the prompt costs nothing, not that it helps.

Together with the zero-loss control in `FINDINGS.md`, where every concept's prompt stays bit-for-bit
identical and the reported score is equal or better, that is four independent ways of removing the
prompt's per-concept content - never training it, zeroing it, deleting it, zeroing the loss - and none
of them costs anything. Whatever carries the method's detection quality, it is not the prompt.

### The same thing in the reference, on both benchmarks and under both protocols

`UCAD_NO_PROMPT` does it in the authors' own code: the scoring backbone is built without e-prompt, so
the prefix tokens are gone from all twelve blocks and it is the same frozen ViT the key comes from.
Nothing is trained. Two readings of that model, in the configuration this repository reproduces the
paper with - ImageNet-21k weights, block 5, bank 196, 224px, VisA on the official 1cls split:

| | MVTec | VisA 1cls |
|---|---|---|
| honest: one bank, mean over 25 seeds | 0.9071 | 0.7810 |
| honest: one bank, **max** over 25 seeds, per category | 0.9356 | 0.8456 |
| **reported protocol, no prompt** (3 seeds) | **0.9295 +- 0.0013** | **0.8706 +- 0.0013** |
| reported protocol, 25 epochs of SCL | 0.9259 | 0.8644 |
| **paper** | **0.930** | **0.874** |

Pixel AUPR follows: 0.4548 with no prompt on MVTec against 0.456 published and 0.4512 with SCL.

And on the other candidate for "closest to the paper" - the ImageNet-1k fine-tune on the per-category
folder copy, the pairing that matches the published VisA *average* while missing every category:

| | MVTec | VisA (folder copy) |
|---|---|---|
| honest, mean over 5 seeds | 0.9170 +- 0.0060 | 0.7933 +- 0.0172 |
| **reported protocol, no prompt** | **0.9412 +- 0.0011** | **0.8762 +- 0.0071** |
| reported protocol, 25 epochs of SCL | 0.9435 | 0.8725 +- 0.0046 |
| paper | 0.930 | 0.874 |

Four cells out of four reproduce the published figures with the prompt deleted, and three of the four
land above the fully trained model. The conclusion does not depend on which configuration you accept
as the paper's.

The middle row of the first table is the cleanest statement of where the published numbers come from.
Twenty-five seeds of the honest configuration - one bank each, no prompt, no gradient - and taking the
best per category, exactly the selection the reported protocol performs, already gives 0.9356 on
MVTec, **above** the published 0.930. Per-category spread across those seeds is what drives it: screw
0.522-0.744, pcb4 0.496-0.845, candle 0.517-0.724, pcb1 0.695-0.883, all from one frozen model whose
only moving part is which 196 vectors the coreset happened to draw. An epoch of this method is a
re-roll, and the protocol reports the best of twenty-five.

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
narrower: the 1960-vector reading is computed and not reported as a detection figure, though its
distance from the 196-vector one is what the published FM turns out to be.

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

- Why block 1 misses by +0.080 on MVTec in Table 7 when blocks 3 to 9 land within 0.005.
- Why the MVTec no-SCL cell at bank 196 misses by +0.021 when the 392 and 784 cells land within 0.002.
- Why pcb4 (-0.066), pcb3 (-0.042) and screw (-0.035) deviate when the other categories agree to 0.01.
- Whether the authors ran a version with the inference phase uncommented.
- What the `train()` method and `get_normal_prototypes_seg` are for. They build k-means prototypes
  over groups of four images, are not on the path `run_ucad.py` takes, would break for a batch size
  that is not a multiple of four, and the paper does not describe them. `prototype_size=5` is passed
  to the model with the comment `# failure version`.
