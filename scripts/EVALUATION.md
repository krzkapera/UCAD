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

Three readings of the same trained model, VisA, twelve categories:

| reading | VisA image AUROC |
|---|---|
| one model - the method as the paper describes it | 0.7045 *(pyCLAD)* |
| mean of all 25 epochs' rescaled scores | 0.8335 *(pyCLAD)* / 0.8287 *(here)* |
| that mean, at the epoch with the best test-set AUROC | 0.8725 *(pyCLAD)* / 0.8723 *(here)* |
| published | 0.874 |

The ensemble is worth **+0.129** and the epoch selection a further **+0.044**. Both are needed to reach
the published figure, and neither appears in the paper. The two implementations agree on the total to
0.0002, which is the strongest evidence we have that we understand what the code does: two independent
codebases, put through the same protocol, land on the same number.

The epoch selection is a leak - the epoch is chosen using the labels of the test set the result is then
reported on. Two symptoms show it is fitting noise rather than finding a stopping point: the epoch it
picks moves between seeds (candle 2/0/3, capsules 10/24/22, cashew 15/23/24), and the metric it does
not optimise gets slightly worse, pixel AUPR falling from 0.3280 to 0.3269.

The ensemble is not a leak - it uses no labels - but it is a different method from the one described.
It stores and averages 25 models' opinions, which costs 25 forward passes over the test set per
concept and, if you wanted to deploy it, 25 banks instead of one.

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
each category's test set instead of 20. It does not: the per-epoch swings within one run reach 0.42 on
both.

## What an honest protocol would look like

- Train. Take **one** model: the last epoch, or an epoch chosen on a validation split disjoint from the
  test set. Evaluate it once.
- Report the spread over at least three seeds. A single VisA category moves by +-0.07 between seeds and
  the twelve-category average by +-0.03, so a difference smaller than that is not a result.
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
