# How we know the contrastive loss contributes nothing

Written for someone who has not seen this project. It sets out what UCAD does, what its code
reports, and the one experiment that settles what the contrastive loss is worth. Every number here
was produced by this repository; `REPRODUCTION.md` says how to produce them.

## The method in one page

UCAD learns a sequence of object categories - "concepts" - one at a time, and must afterwards detect
defects in any of them without being told which category a test image belongs to.

For each concept it stores three things, which the paper calls the key-prompt-knowledge memory (CPM):

- **knowledge**: 196 feature vectors summarising that concept's normal images
- **key**: 196 feature vectors used to recognise which concept a test image belongs to
- **prompt**: a small set of learnable parameters injected into a frozen ViT-B/16

Scoring a test image is PatchCore's procedure. The image becomes a 14x14 grid of patch features. Each
patch's anomaly score is its distance to the **nearest** vector in the concept's knowledge bank, and
the image's score is the maximum over its patches. So the knowledge bank is the whole detector: a
patch that resembles something in the bank is normal, one that resembles nothing in it is anomalous.

The prompt is what the contrastive loss (SCL) trains, for 25 epochs per concept, using SAM
segmentation masks: patch features inside one SAM segment are pulled together, patches in different
segments are pushed apart.

The paper credits SCL with **+0.088 image AUROC on VisA** (Table 5: 0.786 without it, 0.874 with it).
That number is what this document is about.

## What the code reports, and what the paper says about it

The paper's Metrics section says only this:

> we utilize Area Under the Receiver Operating Characteristics (AUROC/AUC) ... For pixel-level
> anomaly segmentation capability, we employ Area Under Precision-Recall (AUPR/AP) ... During the
> inference, we evaluate the model after training on all tasks.

A reader takes that to mean: train the model, then measure it. The code does something else, in three
steps.

**One: it evaluates the test set after every epoch and keeps every result.**

```python
for epoch in range(epochs):
    ...train one epoch...
    PatchCore.prompt_model.eval()
    memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], memory_size)
    PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
    scores, segmentations, labels_gt, masks_gt = PatchCore.predict_prompt(dataloaders["testing"])
    aggregator["scores"].append(scores)
```

**Two: it rescales each epoch's scores and averages all of them.** The reported score of an image is
not one model's score, it is the mean of twenty-five models' scores:

```python
scores = np.array(aggregator["scores"])
min_scores = scores.min(axis=-1).reshape(-1, 1)
max_scores = scores.max(axis=-1).reshape(-1, 1)
scores = (scores - min_scores) / (max_scores - min_scores)
scores = np.mean(scores, axis=0)
```

**Three: of those twenty-five running averages it keeps the one that scores best on the test set.**

```python
if (auroc > pr_auroc):
    memory_feature_list[dataloader_count] = memory_feature
    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
```

`auroc` there is computed against `anomaly_labels`, the test set's ground truth. The state that ends
up in memory, and the number that ends up in the results file, are chosen by looking at the labels of
the data the result is then reported on. A fourth line stops a category the moment it is perfect:

```python
if (auroc == 1):
    break
```

None of this is in the paper, which describes one model per concept and a single evaluation at the
end. On MVTec the early stop fires on five of fifteen categories.

**What an honest protocol looks like.** Train; take *one* model - the last epoch, or an epoch chosen
on a validation split that is disjoint from the test set; evaluate it once; report the spread over
several seeds. If several models are ensembled, say so, and count the memory and compute they cost,
because an ensemble of 25 is a different method from the one the paper describes. Under that protocol
the method scores 0.74-0.80 on VisA as a single model depending on the epoch, against 0.8638 as the
code reports it.

## The experiment: 25 epochs with no loss at all

If SCL earns the +0.088, then removing it while keeping everything else must lose it. So we replaced
the loss with a constant zero:

```python
if variant == 'zero':
    return (similarity_matrix * 0).mean()
```

**The obvious objection is that this cannot do anything.** A zero loss has zero gradient, so
`loss.backward()` writes zeros, `optimizer.step()` adds nothing, and after 25 epochs the prompt is
the prompt it started with. The model never changes. Every epoch should produce an identical score,
and the reported number should be exactly the untrained one.

That objection is right about the model and wrong about the number, and the gap between those two is
the whole finding.

**What differs between epochs is not the model, it is the memory bank.** Look again at step one:
every epoch calls `fit_with_limit_size_prompt`, which re-extracts the training features and
subsamples them to 196 vectors with an approximate greedy coreset. That sampler is random in two
places:

```python
def _reduce_features(self, features):
    mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
```

```python
start_points = np.random.choice(len(features), number_of_starting_points, replace=False)
```

A fresh `Linear` is built on every call, its weights drawn from torch's global generator, and the
starting points from numpy's. Both generators advance as the run proceeds. So epoch 1 and epoch 2
keep **different** 196 vectors out of the same unchanged features, and score the test set slightly
differently.

That is all the machinery needs. Twenty-five differently-subsampled banks give twenty-five different
score vectors; averaging them cancels part of the sampling noise, and the best of twenty-five noisy
readings of one test set is biased upward. Neither operation requires the model to have learned
anything - only that the readings differ from each other.

Result, three seeds, image AUROC:

| | MVTec | VisA |
|---|---|---|
| untrained, one reading | 0.9153 | 0.7872 |
| **zero loss, 25 epochs, the code's protocol** | **0.9271 +- 0.0011** | **0.8708 +- 0.0035** |
| SCL, 25 epochs, the code's protocol | 0.9259 +- 0.0006 | 0.8644 +- 0.0018 |
| paper | 0.930 | 0.874 |

On VisA the two sets of seeds do not overlap: the worst run that learned nothing beats the best run
that learned with SCL. On MVTec the difference is 0.001 with overlapping ranges, which is no
difference. **The +0.088 the paper credits to SCL is reproduced here as +0.084 by a run that performs
no learning at all.**

## Why the paper's ablation cannot see this

Table 5 compares "CPM" against "CPM + SCL". Without SCL there is no loss, so there is no training,
so every iteration produces the same model and the same bank. Averaging twenty-five identical score
vectors is the identity; the maximum of twenty-five equal numbers is that number. Both mechanisms are
switched off in the no-SCL row and switched on in the SCL row.

So the ablation does not compare a loss against no loss. It compares **one reading** against **the
selected mean of twenty-five**. SCL's contribution to that comparison is that it makes the epochs
differ from each other; coreset resampling makes them differ just as much, for free.

## What is left of the method

If the loss contributes nothing, the prompt it trains contributes nothing either, and that is
measurable directly. With SCL off, `reset_prompt` draws from the same seed for every concept, so
every concept's stored prompt is bit identical:

```
PROMPT concept:1 vs_concept0_maxdiff:0.0 cosine:1.0
PROMPT concept:2 vs_concept0_maxdiff:0.0 cosine:1.0
```

With SCL on, after three epochs they differ by a cosine of 0.9995 - directionally the same vector.

The key works: with every concept in memory it routes every test image to its own concept on both
benchmarks, and the routed reading equals the each-concept-against-its-own-bank reading to four
decimals. But it is not needed. One bank holding every concept's vectors, at the same total memory
and no routing at all, scores the same or better:

| untrained | per-concept banks, routed | one bank holding all concepts |
|---|---|---|
| MVTec | 0.9153 / 0.4255 | 0.9154 / 0.4223 |
| VisA | 0.7872 / 0.2455 | 0.7971 / 0.2435 |

Stored memory is equal - fifteen banks of 196 either way. What differs is the query: the shared bank
searches fifteen times more vectors per patch, while the routed one first compares the image against
fifteen key banks, so the compute is comparable rather than identical.

So of key-prompt-knowledge: **the prompt is inert, the key is redundant, and the knowledge is a
PatchCore coreset that is never discarded.** CPM's measured value is that last part alone - it is the
bank the anomaly score is computed against, and keeping one per concept instead of overwriting it is
what the +0.20 of the first ablation row buys. Forgetting is zero because nothing is shared between
concepts, which is a property of that arrangement rather than a result.

## Supporting evidence

**Training makes the single model worse.** On VisA a single model scores 0.787 untrained, peaks near
0.80 somewhere in the first few epochs - epoch 0, 2, 3 and 7 across four runs, so the peak is not a
property of the schedule - and falls to 0.73-0.76 by epoch 25. A 100-epoch run on three categories
keeps falling to the end.

**The loss has no equilibrium.** `-cos` on same-segment pairs is minimised when a segment collapses
to a single point; `exp(cos)` on different-segment pairs when segments are maximally spread; nothing
anchors the features to where they started. Its optimum is a degenerate embedding. A defect sits
*inside* a segment, so collapsing segments removes the very variation the nearest-neighbour score
reads.

**The geometry moves exactly that way.** Measured after each epoch on VisA and averaged over the 12
categories, the mean cosine between patches of the same SAM segment rises 0.421 -> 0.582 over 15
epochs while between segments it falls 0.181 -> 0.124, both monotonically, and image AUROC peaks at
epoch 1. Followed further on one category, macaroni2 reaches 0.858 within-segment cosine by epoch 86,
so the drift does not level off - it is heading for the collapse the loss is minimised by. Per category the correlation between the two is negative in 10 of
12, from -0.17 to -0.58. The drift is unambiguous; its link to the quality drop is an association,
not an isolated cause - one category loses quality without collapsing at all.

**It is not a slip in how the loss was coded.** The code exponentiates the negative pairs and divides
by a temperature; the paper's Eq. 3 does neither. Writing Eq. 3 literally recovers 0.005 on VisA and
leaves the shape unchanged: up for two or three epochs, down thereafter.

## What this does not say

It does not say the published numbers are wrong: they reproduce closely, including both ablation
tables - see `REPRODUCTION.md`. It does not say SCL is harmful in general, only that on these two
benchmarks in this configuration it costs 0.006 image AUROC on VisA and nothing on MVTec, while the
protocol it is measured under credits it with 0.088. And it does not establish the mechanism of the
damage beyond the association described above.
