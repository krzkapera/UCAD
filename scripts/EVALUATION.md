# The reporting protocol, taken apart

`FINDINGS.md` shows that the protocol this code reports under is what makes the contrastive loss look
useful. This file takes the protocol apart: what each of its two mechanisms is worth, what happens if
you replace them with something defensible, and why forgetting cannot be measured by a run of this
code at all.

Two implementations were used. Unless a row says otherwise, numbers are from this repository with
ViT-B/16 ImageNet-21k weights. Rows marked *pyCLAD* come from an independent reimplementation of the
method in a continual-learning library, which uses timm's current default weights for
`vit_base_patch16_224` - an ImageNet-1k fine-tune, not the checkpoint this code configures - and so are
comparable among themselves but not directly against the rows above them.

## The two mechanisms, separately

Three readings of the same trained model, VisA, twelve categories. **This decomposition was measured
before we knew which split and checkpoint the paper used, so every row of it is on the per-category
folder copy with timm's default ImageNet-1k-fine-tuned weights** - which is also why its bottom row
sits on the published figure. The rows are comparable to each other and not to anything on the
official split.

| reading | VisA image AUROC |
|---|---|
| one model - the method as the paper describes it | 0.7045 *(pyCLAD)* |
| mean of all 25 epochs' rescaled scores | 0.8335 *(pyCLAD)* / 0.8287 *(here)* |
| that mean, at the epoch with the best test-set AUROC | 0.8725 *(pyCLAD)* / 0.8723 *(here)* |
| published | 0.874 |

The ensemble is worth **+0.129** and the epoch selection a further **+0.039**, both measured in pyCLAD,
which is the only one of the two where a single model can be read at all - this code has no way to
report one. In this code the selection alone is worth **+0.0436**, from 0.8287 to 0.8723. Neither
mechanism appears in the paper, and both are needed to reach the published figure. The two
implementations agree on the total to 0.0002, which is the strongest evidence we have that we
understand what the code does: two independent codebases, put through the same protocol, land on the
same number.

On the official split with the checkpoint the paper states, we have the two ends of the same
decomposition but not its middle, because the code only ever reports the selected ensemble: a single
model at its last epoch scores 0.7420 and 0.7314 over two seeds, against 0.8638 and 0.8668 for the
selected ensemble, and 0.7872 untrained. The middle row would need the unselected ensemble, which this
code does not write.

The epoch selection is a leak - the epoch is chosen using the labels of the test set the result is then
reported on. Two symptoms show it is fitting noise rather than finding a stopping point, both measured
on the folder copy: the epoch it picks moves between seeds (candle 2/0/3, capsules 10/24/22, cashew
15/23/24), and the metric it does not optimise gets slightly worse, pixel AUPR falling from 0.3280 to
0.3269.

The ensemble is not a leak - it uses no labels - but it is a different method from the one described.
It stores and averages 25 models' opinions, which costs 25 forward passes over the test set per
concept and, if you wanted to deploy it, 25 banks instead of one.

## Mean or maximum? Both, nested

The two mechanisms are easy to confuse because they are layered. Per concept, per epoch, the code:

1. scores the whole test set with this epoch's model and appends the score vector to `aggregator`;
2. rescales **every** epoch's vector collected so far to 0..1 and averages them - a *running* mean over
   epochs 1..k, not this epoch alone;
3. computes the image AUROC of that running mean;
4. if it is the best such AUROC so far, stores this epoch's prompt and bank and records the number.

So the reported figure is the **maximum over epochs of the AUROC of the cumulative mean**. A maximum of
means. The averaging smooths, the maximum then picks the luckiest point of the smoothed sequence, and
because both operate on the same 25 readings, the two effects compound.

One consequence is worth stating on its own: what a run *stores* is a single epoch's prompt and bank -
the epoch at which the running mean peaked - while what it *reports* is the mean of 25 epochs. The
artefact cannot reproduce the number. We measured the gap: replaying the stored state through the
routing phase gives 0.7801 on VisA against the 0.8638 the same run reports.

## Why the protocol looks like this

The averaging is not a design decision about epochs. It is PatchCore's ensemble code with an epoch loop
wrapped around it, and the file shows this directly. In the (commented-out) inference phase the
identical block reads:

```python
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                ...
                aggregator["scores"].append(scores)
            scores = np.array(aggregator["scores"])
            ...
            scores = np.mean(scores, axis=0)
```

That is the original and legitimate pattern: PatchCore supports several backbones as an ensemble, and
this averages their min-max rescaled scores - the rescaling is there precisely because different
backbones score on different scales. In the training phase the same three pieces appear, but the
aggregator is initialised **before** the epoch loop while the averaging sits **inside** it:

```python
            aggregator = {"scores": [], "segmentations": []}      # before the loop
            ...
            for epoch in range(epochs):
                for i, PatchCore in enumerate(PatchCore_list):
                    ...
                    aggregator["scores"].append(scores)
                scores = np.array(aggregator["scores"])           # inside the loop
```

A list meant to hold one entry per ensemble member now holds one entry per epoch. And since every UCAD
run uses a single backbone, `PatchCore_list` has length one, so the original averaging was a no-op -
wrapping the epoch loop around it turned that no-op into a 25-member ensemble.

The epoch selection has a more ordinary origin. `if(auroc>pr_auroc): ...store the state...` is the
standard "keep the best checkpoint" idiom, which is correct and universal when the metric comes from a
validation split. Here it comes from the test set. It is a one-line mistake about which set you monitor,
and the `if(auroc==1): break` beside it is the same mistake applied to how long you train.

## Does any of it make sense?

**The averaging: as a technique yes, as a reported result no.** Averaging the predictions of checkpoints
taken along one training run is a real method - snapshot ensembling - and it does what it did here,
reduce variance. Two things make it indefensible as reported. It is not described, so a reader compares
it against baselines that report one model. And the paper's own memory accounting - 23.28MB for one key,
one prompt and one knowledge bank per concept - is the accounting for a single model, while the number
beside it needs 25 banks to reproduce.

**The selection: no.** Choosing anything with the labels of the set you report on inflates the result by
an amount that grows with how noisy the readings are, which is why it is worth more here (+0.04) than it
would be for a stable method. The metric it does not optimise moves the other way, which is the usual
symptom.

**The early break: it saves compute and costs credibility.** Stopping when test AUROC hits exactly 1.0
means the amount of training is also chosen from test labels.

The honest version of this protocol is not complicated: keep the ensemble if you want it, describe it,
count its memory, and choose the stopping point on a validation split. We measured what that costs -
see the table above - and the answer is most of the published margin.

## Replacing the leak with something defensible

If the epoch has to be chosen, choose it on data the result is not reported on. We measured that, and
several criteria that need no anomaly labels at all, on both VisA splits. Each criterion picks one
epoch per concept; all of them are then scored on the same held-out half of the test set, so the
columns are comparable. Three seeds, *pyCLAD*, single model - no ensemble.

| criterion | VisA folder copy | VisA official split |
|---|---|---|
| best epoch on the held-out half - unreachable upper bound | 0.8955 | 0.8836 |
| epoch chosen on the test labels - the leak | 0.8735 | 0.8659 |
| **epoch chosen on a labelled validation half** | **0.8123** | **0.8316** |
| **no training at all** | **0.8145** | **0.8381** |
| last epoch | 0.7403 | 0.7755 |
| best label-free criterion | 0.7886 | 0.8320 |

The honest replacement works - it is worth +0.07 and +0.06 over taking the last epoch - and it does not
recover the published number: the leak is worth a further 0.061 and 0.034 on top of it.

More importantly, **honest early stopping does not beat not training at all** on either split: 0.8123
against 0.8145, and 0.8316 against 0.8381. On MVTec it does, by +0.027, and that gain is concentrated
in three of fifteen categories - screw +0.257, cable +0.074, hazelnut +0.049 - with the other twelve
moving by less than 0.02.

The label-free criteria were: where held-out normals land (mean, max, 95th percentile), the false
positive rate of one half of them against a threshold from the other, the mean pairwise cosine and the
effective rank of the stored bank, and a two-means split of the unlabelled test scores. All of them
land below the untrained model on both splits. They measure the wrong thing: the loss changes the
feature geometry monotonically with training while detection quality moves up and down inside that
trend, so a criterion that tracks the geometry tracks the epoch count.

We expected the official split to calm the epoch-to-epoch noise, since it puts 100 normal images in
each category's test set instead of 20, and it partly does. Measured in this implementation, the
largest within-category swing across the 25 epochs of one run falls from 0.33-0.41 on the folder copy
to 0.23-0.24 on the official split, and the median category's swing from 0.25-0.29 to 0.16-0.17. So
roughly a third of the instability was the estimate rather than the model - and the remaining two
thirds are enough to leave every conclusion in the table above unchanged. An earlier version of this
file said the noise did not fall at all, which was wrong: it was comparing a folder-copy figure
against itself.

## What an honest protocol would look like

- Train. Take **one** model: the last epoch, or an epoch chosen on a validation split disjoint from the
  test set. Evaluate it once.
- Report the spread over at least three seeds. On the folder copy a single VisA category moves by
  +-0.07 between seeds and the twelve-category average by +-0.03; on the official split the average
  moves by +-0.002 over two runs of the same configuration. A difference smaller than the spread you
  measure is not a result.
- If models are ensembled, say so, and count what they cost - an ensemble of 25 is not the method the
  paper describes.
- Never choose anything using the labels of the set you report on.

Under that protocol, on the official split with the checkpoint the paper states, the method scores
0.7801 on VisA and 0.9189 on MVTec as a single model after the whole sequence, against 0.7872 and
0.9153 for the same thing untrained.

## Forgetting

The Forgetting Measure (Eq. 7) is defined over the matrix of "concept j's score after concept k was
learned", for every k >= j. **A run of the released code cannot produce that matrix.** The phase that
would - route a test image by its key, retrieve that concept's prompt and knowledge, evaluate every
concept once all of them are learned - sits inside a triple-quoted string in `run_ucad.py` and never
executes. `results.csv` comes from the training loop, where each concept is evaluated immediately after
it is learned and never revisited.

`UCAD_LOG_FM=1` produces the matrix. It is constant along its rows:

```
FM_MATRIX learned:0 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:1 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:1 eval:1 routed:1 auroc:0.715704647676162
FM_MATRIX learned:2 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:2 eval:1 routed:1 auroc:0.715704647676162
```

| | tasks | misrouted images | avg FM | paper |
|---|---|---|---|---|
| MVTec, 0 epochs | 15 | 0 | 0.000000 | - |
| MVTec, 25 epochs | 15 | 0 | 0.000000 | 0.010 |
| VisA, 0 epochs | 12 | 0 | 0.000000 | - |
| VisA, 25 epochs | 12 | 0 | 0.000000 | 0.039 |

Exactly zero, not approximately. It follows from the architecture: nothing is shared between concepts,
so the only thing that could move an earlier concept's score is the key routing elsewhere, and it never
does - `routed` equals `eval` on all 1725 MVTec and 2162 VisA test images. The same holds in the
independent implementation, which reports 0.0000 and 100% routing on both benchmarks.

So the published 0.010 and 0.039 have no source in this code or in this design, and the honest number
for a method with per-concept memory and exact routing is 0. That is worth stating plainly rather than
as a strength: zero forgetting here is a property of keeping the concepts in separate boxes, not a
result about learning.

One caveat if you compute this with the pyCLAD library rather than the matrix above: its
`ForgettingMeasure` averages over `range(learned_task + 1)`, so it includes the concept just learned
and compares it against readings taken before that concept was in memory - readings that are
near-random for per-concept memory. It returns -0.007 to -0.027 where Eq. 7 returns 0. Its
`BackwardTransfer` agrees with Eq. 7 for this model.
