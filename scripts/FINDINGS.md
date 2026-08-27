# What the contrastive loss is worth

Written for someone who has not seen this project. It sets out what UCAD does, what its code reports,
and what the contrastive loss actually contributes, which is nothing - including after the one
defect that could plausibly explain it is repaired. Every number here was produced by this
repository; `REPRODUCTION.md` says how to produce them.

The short version, if you read no further:

- As released, switching the loss off changes nothing measurable on either benchmark, at any prompt
  length, and the published gain of +0.088 on VisA is the reporting protocol rather than the loss.
- The loss is not broken and is not miscoded: it moves the features exactly where it says it will.
- There is a real design defect behind it - the loss optimises cosine geometry while the bank is
  searched by Euclidean distance over vectors whose lengths nothing constrains, and those lengths
  have a spread 1.5 to 1.8 times their mean. Repairing it in one line is worth +0.015 to +0.030 on
  its own, more than all of the training.
- **Repairing it does not rescue the loss.** With the geometries aligned, a single trained model
  still scores below its own untrained control: 0.7846 against 0.8118 on VisA over disjoint seed
  ranges. The loss wins only under the reported protocol, and that is an ensembling effect - it makes
  the 25 averaged epochs more diverse, while each of them is worse.

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

**One: it evaluates the test set after every epoch and keeps every result.** Condensed from
`run_ucad.py` lines 160-191 - the interleaved second evaluation at `basic_size` is left out:

```python
for epoch in range(epochs):
    ...train one epoch...
    PatchCore.prompt_model.eval()
    memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], memory_size)
    PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
    scores, segmentations, labels_gt, masks_gt = PatchCore.predict_prompt(
        dataloaders["testing"]
    )
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
if(auroc>pr_auroc):
    memory_feature_list[dataloader_count] = memory_feature
    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
```

`auroc` there is computed against `anomaly_labels`, the test set's ground truth. The state that ends
up in memory, and the number that ends up in the results file, are chosen by looking at the labels of
the data the result is then reported on. A fourth line stops a category the moment it is perfect:

```python
if(auroc==1):
    break
```

None of this is in the paper, which describes one model per concept and a single evaluation at the
end. How often the early stop fires depends on the checkpoint: with the one the paper states it fires
on two of fifteen MVTec categories, bottle and leather; with timm's ImageNet-1k fine-tune, which puts
five categories at exactly 1.0, it fires on five.

**What an honest protocol looks like.** Train; take *one* model - the last epoch, or an epoch chosen
on a validation split that is disjoint from the test set; evaluate it once; report the spread over
several seeds. If several models are ensembled, say so, and count the memory and compute they cost,
because an ensemble of 25 is a different method from the one the paper describes. Under that protocol
the method scores 0.73-0.80 on VisA as a single model depending on which epoch you take, against
0.8638 as the code reports it.

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
every epoch calls `fit_with_limit_size_prompt`, which re-extracts the training features and hands them
to the sampler,

```python
features = self.featuresampler.run_with_limit_memory(features, limit_size)
```

which reduces them and then picks 196 of them greedily. It is random in two places:

```python
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
```

```python
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()
```

The features are 1024-dimensional and `dimension_to_project_features_to` is 128, so the early return
never fires: a fresh `Linear` is built on every call, its weights drawn from torch's global generator,
and the greedy search's starting points from numpy's. Both generators advance as the run proceeds. So
epoch 1 and epoch 2 keep **different** 196 vectors out of the same unchanged features, and score the
test set slightly differently.

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

If the loss changes nothing as released, the prompt it trains carries nothing either, and that is
measurable directly. With SCL off, `reset_prompt` draws from the same seed for every concept, so
every concept's stored prompt is bit identical:

```
PROMPT concept:1 vs_concept0_maxdiff:0.0 cosine:1.0
PROMPT concept:2 vs_concept0_maxdiff:0.0 cosine:1.0
```

With SCL on, after three epochs they differ by a cosine of 0.9995 - directionally the same vector.

The key works, though it is asked an easier question than the paper describes. Eq. 4 routes an
individual image; the code sums the anomaly scores of a concept's **entire test set** against each
candidate key bank and takes the argmin, so one decision covers the whole set - fifteen decisions on
MVTec, twelve on VisA. All of them are correct, and the routed reading equals the
each-concept-against-its-own-bank reading to four decimals. Per-image routing, as the paper defines it,
is also exact, but that was measured in the independent implementation rather than here. But it is not needed. One bank holding every concept's vectors, at the same total memory
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
leaves the shape unchanged: up for two or three epochs, down thereafter. Normalising each term of
Eq. 3 by its own pair count, which equal-weight sums do not, changes nothing either: 0.9251 on MVTec
and 0.8668 on VisA against 0.9259 and 0.8644 for the released form.

**Nor does the loss fail at its own objective. It succeeds at it.** Followed per epoch on VisA's
candle, within-segment cosine rises 0.5197 -> 0.6957 and between-segment falls +0.3477 -> -0.0751 over
25 epochs, monotonically. Eq. 3 asks for exactly that and gets it.

## Why it succeeds and still does not help

**The objective and the score are measured in different geometries.** The loss L2-normalises before it
measures anything, so it constrains directions only. The bank is a plain `faiss.IndexFlatL2` over
unnormalised vectors, and `||a-b||^2 = ||a||^2 + ||b||^2 - 2||a|| ||b|| cos`. Nothing constrains the
norms, and they are not a minor term: measured on the features that go into the bank, their standard
deviation is 1.8x their mean on MVTec's bottle (227 against 125) and 1.5x on VisA's candle (101
against 69). The distance the score reads is dominated by length, and length is the one thing the loss
never sees. Over 25 epochs on candle the norm drifts 69.0 -> 97.6 while its spread falls to 53 by
epoch 15 and rebounds to 130 - moved around freely, because nothing holds it.

**Aligning the two geometries is worth a great deal - but not to the loss.**
`UCAD_NORMALIZE_FEATURES=1` L2-normalises the features on the way into the bank and the query, which
makes the L2 index a cosine index. Three seeds each, reported protocol, ImageNet-21k weights, VisA on
the official split:

| | untrained | SCL, 25 epochs | zero loss, 25 epochs |
|---|---|---|---|
| MVTec, as released | 0.9153 | 0.9259 | 0.9271 |
| MVTec, normalised | 0.9306 +- 0.0044 | 0.9455 +- 0.0013 | 0.9466 +- 0.0001 |
| VisA, as released | 0.7872 | 0.8644 | 0.8708 |
| VisA, normalised | 0.8052 +- 0.0043 | 0.8944 +- 0.0018 | 0.8857 +- 0.0010 |

The normalisation alone is worth +0.015 to +0.030, more than everything the paper's training does,
and the untrained normalised model on MVTec already matches the published 0.930.

The VisA cell where SCL beats its control by +0.0087 is the only one in this whole analysis where it
wins - and it is an artefact of the column it sits in. Every number in that table is under the
reported protocol, so both sides get 25 coreset re-rolls, the ensemble and the selection. Read the
same runs as single models instead, at the last epoch, which is the honest comparison:

| normalised, single model at epoch 25 | SCL | zero loss |
|---|---|---|
| VisA image AUROC | 0.7846 +- 0.0164 | **0.8118 +- 0.0124** |
| VisA pixel AUPR | 0.2623 +- 0.0154 | 0.2577 +- 0.0157 |
| MVTec image AUROC | 0.8991 +- 0.0108 | **0.9058 +- 0.0018** |
| MVTec pixel AUPR | 0.4400 +- 0.0060 | 0.4333 +- 0.0115 |

**SCL loses on both benchmarks**, and on VisA the seed ranges are disjoint (0.7671-0.7996 against
0.8042-0.8261). It loses at every honest reading of the same runs, not only the last epoch:

| normalised, VisA / MVTec | SCL | zero loss |
|---|---|---|
| last epoch | 0.7846 / 0.8991 | **0.8118** / **0.9058** |
| best epoch common to all concepts, chosen on the test labels | 0.8208 / 0.9279 | **0.8292** / **0.9294** |
| best epoch per concept, chosen on the test labels | **0.8627** / 0.9518 | 0.8587 / **0.9539** |
| the reported protocol | **0.8944** / 0.9455 | 0.8857 / **0.9466** |

On MVTec the best epoch is 0 or 1 in all three runs, so training makes it worse from the first step.

What the protocol column is measuring instead is ensemble diversity. It averages 25 epochs; with SCL
the prompt moves, so those 25 members differ from one another in more than their coreset draw, and a
more diverse ensemble averages better. The members themselves are worse. That reading is an inference
from the two tables rather than a direct measurement, and it is the one thing here still worth
testing directly.

So the geometry mismatch is real, and repairing it is the largest single improvement in this
document - but the loss does not become useful once it is repaired. It halves its own damage on VisA,
from -0.045 to -0.027 against the untrained model, and stays negative.

**The cross-image half of ReConPatch makes it worse, not better.** The paper names ReConPatch as its
inspiration, but ReConPatch forms positive pairs from feature-space nearest neighbours across the
training set while Eq. 3 forms them from segment identity inside a single image, so no pair ever leaves
its image. Adding the missing half - the k nearest patches in other images of the batch count as
positives too - costs 0.0123 on VisA under normalisation (0.8821 +- 0.0019 against 0.8944 +- 0.0018,
disjoint) and changes nothing on MVTec. The intra-image restriction the paper imposes turns out to be
the right call.

## Four things that are not the explanation

Each of these looks like it could be the bug behind "the loss does nothing", and each was checked and
is clean. They are recorded so nobody spends a day re-deriving them as suspicions.

**The optimisation runs.** The loss falls monotonically over 25 epochs, and far: on VisA's first
concept from 0.970 to 0.088, on another down to -1.23. Nothing is stuck or diverging.

**The SAM label maps are read correctly.** `forward_head` takes channel 0 of `cv2.imread`'s output,
which would merge distinct segments if the maps were colour-coded. The masks in `mvtec2d-sam-b.zip`
are PNG colour type 0 - 8-bit greyscale - so `imread` replicates the one channel three times and
channel 0 is the segment id. `scripts/visa_sam_b_masks.py` writes VisA's the same way.

**The optimiser is what the paper says.** From `args_dict.npy`: adam, betas (0.9, 0.999), eps 1e-8,
**weight decay 0.0**, clip_grad 1.0, constant schedule. The `lr` stored in that file is 0.02, but
`run_ucad.py` overwrites it with 5e-4 before building the optimiser - the figure in the paper's text
rather than the 5e-5 in its appendix table.

**The prompt is not being reinitialised mid-run by accident.** Each concept builds a fresh
`PatchCore`, and `fix_seeds(seed)` runs first, so every concept starts from the same random prompt.
That is why with the loss off the stored prompts come out bit-for-bit identical across concepts: a
property of the method, not a fault in the measurement.

## Two objections, both closed

**Prompt capacity was not the constraint.** The measurements above are at `prompt_length=1`, and the
paper's own memory accounting implies 7, so the obvious objection is that the loss never had room to
work. It did not need room. Three seeds per cell, reported protocol:

| | SCL | zero loss | difference |
|---|---|---|---|
| MVTec, length 1 | 0.9259 | 0.9271 | -0.0012 |
| MVTec, length 5 | 0.9280 +- 0.0014 | 0.9264 +- 0.0019 | +0.0016 |
| MVTec, length 7 | 0.9290 +- 0.0024 | 0.9289 +- 0.0043 | +0.0001 |
| VisA, length 1 | 0.8644 | 0.8708 | -0.0064 |
| VisA, length 5 | 0.8632 +- 0.0047 | 0.8676 +- 0.0037 | -0.0044 |
| VisA, length 7 | 0.8694 +- 0.0038 | 0.8683 +- 0.0021 | +0.0011 |

Every difference sits inside the seed spread, at every capacity, on both benchmarks.

**Half the prompt is never trained**, which does not rescue the loss but does change what the method
is. `e_prompt_layer_idx` inserts prefix tokens at all twelve blocks, but the loss is computed on
`res['seg_feat']`, captured after block 5, and `res['x']` after the last block enters no objective.
`batched_prompt` is a plain index into one `nn.Parameter` with no coupling across layers, so the
slices for blocks 6 to 11 receive exactly zero gradient and stay at their random initialisation for
the whole run. The trainable capacity is six layers, not twelve, and the memory the paper accounts for
includes layers that never move.

## What this does not say

It does not say the published numbers are wrong: they reproduce closely, including both ablation
tables - see `REPRODUCTION.md`. It does not say SCL is useless in general - normalise the features and
it earns +0.009 image and +0.025 pixel on VisA, repeatably. It says that as released, on these two
benchmarks, it costs 0.006 image AUROC on VisA and nothing on MVTec, while the protocol it is measured
under credits it with 0.088. And it does not establish the mechanism beyond what the geometry log
shows: the loss moves the features the way it promises, in a geometry the scorer does not read.
