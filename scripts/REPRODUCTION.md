# Reproducing the paper

Every table below was produced by this repository. The configuration is the one the paper states and
this code implements; the three settings that are easy to get wrong are called out under
"Configuration" and each is worth more than the method's headline claim.

Numbers are image AUROC and pixel AUPR, averaged over the benchmark's categories. Where a number
rests on fewer than three seeds it says so.

## What you need

**MVTec AD.** Unpack it and rename the directory to `mvtec2d`, the name the loader expects.

**VisA on its official split.** This matters: the loader walks directories and cannot read VisA's
`split_csv/1cls.csv`, so it is on you to materialise the official split as folders. The per-category
copy that circulates - all normals in `train/good` except the first 20 per class - is *not* that
split, and using it costs 0.06 image AUROC and turns one category (candle) into noise. Build the tree
from the csv:

```
<root>/<category>/train/good/<name>.JPG      rows with split=train
<root>/<category>/test/good/<name>.JPG       rows with split=test, label=normal
<root>/<category>/test/bad/<name>.JPG        rows with split=test, label=anomaly
<root>/<category>/ground_truth/bad/<stem>.png
```

Symlinks are fine, but make them absolute - a relative target resolves against the link's own
directory, not your working directory.

**SAM masks.** MVTec's are in `mvtec2d-sam-b.zip` in this repository: 224x224, 8-bit, one per training
image, named exactly like the image (`0020.JPG`, not `.png`) because the code finds them by string
substitution on the image path. VisA's you generate with `dataset_sam.py --sam_type vit_b` and
convert to the same form with `scripts/visa_sam_b_masks.py`. Masks are only needed for training
images; a run with `--epochs_num 0` needs none.

**Environment.** `environment.yaml` pins python 3.8, torch 1.12, timm 0.6.7. On a newer stack use
`scripts/launch.py`, which restores the NumPy 1.x aliases the code still reaches for and caps the
process's address space.

## Configuration

| | value | why it is easy to get wrong |
|---|---|---|
| backbone weights | ViT-B/16 **ImageNet-21k**, no ImageNet-1k fine-tune | `default_cfgs` has the augreg line commented out and `imagenet21k/ViT-B_16.npz` in its place. timm's modern default for the same model name is an ImageNet-1k fine-tune, which scores up to 0.07 higher on VisA. The paper says 21k. |
| VisA split | official `1cls.csv` | see above, worth 0.06 |
| feature block | 5 of 12 | hardcoded in `forward_features`; the paper's Table 7 sweeps it and keeps 5 "for simplicity" |
| bank per concept | 196 | `--memory_size 196`; the code also computes a 1960-vector reading into `results_nolimit/` that the paper never mentions |
| patch neighbourhood | `--patchsize 1` | the CLI default is 3 and it is not harmless: leaving it out costs 0.27 image AUROC on MVTec and 0.25 on VisA at these settings, and neither training nor the epoch ensemble recovers it |
| scorer neighbours | `--anomaly_scorer_num_nn 1` | inert - `PatchCore.load` spells the parameter `anomaly_score_num_nn` and absorbs the other into `**kwargs`, so the scorer always uses 1 |
| input | 224, resize then centre crop | |
| epochs / batch / lr | 25 / 8 / 5e-4, constant schedule | the paper's text says 5e-4, its appendix table says 5e-5 |

The `-b wideresnet50 -le layer2 -le layer3` in the README command does nothing: `PatchCore.load`
builds the ViT unconditionally.

## Running it

```bash
python3 scripts/launch.py --gpu 0 --seed 0 --memory_size 196 --epochs_num 25 \
    --log_group FULL --log_project UCAD "$RUN_DIR/results" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 $D_FLAGS --num_workers 2 mvtec "$DATA_ROOT"
```

`$D_FLAGS` is one `-d <category>` per category; the `mvtec` positional selects the loader, which
reads the VisA tree too. `scripts/run_benchmark.sbatch` wraps this for Slurm.

Results land in `results/UCAD/FULL/results.csv` (the 196-vector bank, this is the reported one) and
`results_nolimit/UCAD/FULL/results.csv` (1960 vectors). Each file holds one row per category and a
single `Mean` row; that row is the benchmark average.

**A run takes** 1.5-1.9 h for MVTec and 2.5-3.0 h for VisA at 25 epochs on one GPU, and 9-11 minutes
at `--epochs_num 0`. `UCAD_CKPT_DIR` saves each finished concept so a job that runs out of wall time can
be resubmitted and continue.

**What the reported number is.** Not one model: the code evaluates the test set after every epoch,
averages the min-max rescaled scores of all epochs so far, and keeps the epoch whose test-set image
AUROC is highest. `FINDINGS.md` sets that out with the code. To read a single model instead, run with
`UCAD_LOG_EPOCHS=1` and take the `SINGLE_EPOCH` lines.

## Results

### Headline

| | paper | here |
|---|---|---|
| MVTec, image AUROC | 0.930 | 0.9259 +- 0.0006 (3 seeds) |
| MVTec, pixel AUPR | 0.456 | 0.4512 (3 seeds) |
| VisA, image AUROC | 0.874 | 0.8638 and 0.8668 in two runs |
| VisA, pixel AUPR | 0.300 | 0.2982 and 0.2999 |

The two VisA runs are the same configuration and seed; one was resumed from per-concept checkpoints
after hitting a wall-time limit, which puts the random number generator in a different state, so they
are effectively two samples. The per-category table below is the first of them.

### MVTec AD, per category, three seeds

| category | image AUROC | paper | pixel AUPR | paper |
|---|---|---|---|---|
| bottle | 0.9995 +- 0.0005 | 1.000 | 0.7509 +- 0.0019 | 0.752 |
| cable | 0.7258 +- 0.0099 | 0.751 | 0.2857 +- 0.0340 | 0.290 |
| capsule | 0.8595 +- 0.0068 | 0.866 | 0.3463 +- 0.0065 | 0.349 |
| carpet | 0.9648 +- 0.0029 | 0.965 | 0.6224 +- 0.0042 | 0.622 |
| grid | 0.9368 +- 0.0048 | 0.944 | 0.1850 +- 0.0022 | 0.187 |
| hazelnut | 0.9950 +- 0.0028 | 0.994 | 0.5142 +- 0.0029 | 0.506 |
| leather | 1.0000 +- 0.0000 | 1.000 | 0.3386 +- 0.0013 | 0.333 |
| metal_nut | 0.9881 +- 0.0015 | 0.988 | 0.7707 +- 0.0080 | 0.775 |
| pill | 0.8785 +- 0.0120 | 0.894 | 0.6241 +- 0.0193 | 0.634 |
| screw | 0.7039 +- 0.0200 | 0.739 | 0.1621 +- 0.0315 | 0.214 |
| tile | 0.9980 +- 0.0002 | 0.998 | 0.5356 +- 0.0065 | 0.549 |
| toothbrush | 0.9981 +- 0.0016 | 1.000 | 0.2922 +- 0.0018 | 0.298 |
| transistor | 0.8993 +- 0.0152 | 0.874 | 0.3907 +- 0.0160 | 0.398 |
| wood | 0.9918 +- 0.0005 | 0.995 | 0.5485 +- 0.0066 | 0.535 |
| zipper | 0.9498 +- 0.0027 | 0.938 | 0.4014 +- 0.0180 | 0.398 |
| **average** | **0.9259** | **0.930** | **0.4512** | **0.456** |

Mean absolute deviation 0.0089 on both metrics; ten of fifteen categories within 0.01 on image AUROC
and twelve of fifteen on pixel AUPR. The one real outlier is screw, low in both.

### VisA, per category, official split, one seed

| category | image AUROC | paper | pixel AUPR | paper |
|---|---|---|---|---|
| candle | 0.7933 | 0.778 | 0.0701 | 0.067 |
| capsules | 0.8558 | 0.877 | 0.4172 | 0.437 |
| cashew | 0.9656 | 0.960 | 0.5859 | 0.580 |
| chewinggum | 0.9486 | 0.958 | 0.4829 | 0.503 |
| fryum | 0.9398 | 0.945 | 0.3364 | 0.334 |
| macaroni1 | 0.8363 | 0.823 | 0.0129 | 0.013 |
| macaroni2 | 0.6543 | 0.667 | 0.0072 | 0.003 |
| pcb1 | 0.9113 | 0.905 | 0.7188 | 0.702 |
| pcb2 | 0.8637 | 0.871 | 0.1570 | 0.136 |
| pcb3 | 0.7711 | 0.813 | 0.2612 | 0.266 |
| pcb4 | 0.8354 | 0.901 | 0.0616 | 0.106 |
| pipe_fryum | 0.9908 | 0.988 | 0.4669 | 0.457 |
| **average** | **0.8638** | **0.874** | **0.2982** | **0.300** |

Mean absolute deviation 0.0172 and 0.0127; nine of twelve categories within 0.02 in both metrics.
On the folder copy instead of the official split the same comparison gives 0.080, and candle alone
lands at 0.3498 against 0.778.

### Table 5, the module ablation

| | MVTec paper | MVTec here | VisA paper | VisA here |
|---|---|---|---|---|
| no CPM, no SCL | 0.693 / 0.183 | 0.6692 / 0.1621 | 0.584 / 0.050 | 0.5862 / 0.0491 |
| CPM, no SCL | 0.894 / 0.426 | 0.9153 / 0.4255 | 0.786 / 0.251 | 0.7872 / 0.2455 |
| CPM and SCL | 0.930 / 0.456 | 0.9259 / 0.4512 | 0.874 / 0.300 | 0.8638 / 0.2982 |

The first row needs `UCAD_INFERENCE=1 UCAD_NO_CPM=1 --epochs_num 0`: one knowledge base that each
task overwrites, so every task is scored against the last task's bank.

### Table 6, the knowledge-size ablation

| bank | MVTec no SCL | MVTec SCL | VisA no SCL | VisA SCL |
|---|---|---|---|---|
| 196, paper | 0.894 / 0.426 | 0.930 / 0.456 | 0.786 / 0.251 | 0.874 / 0.300 |
| 196, here | 0.9153 / 0.4255 | 0.9259 / 0.4512 | 0.7872 / 0.2455 | 0.8638 / 0.2982 |
| 392, paper | 0.921 / 0.452 | 0.936 / 0.461 | 0.818 / 0.255 | 0.893 / 0.307 |
| 392, here | 0.9203 / 0.4485 | 0.9401 / 0.4614 | 0.8315 / 0.2716 | 0.8852 / 0.3069 |
| 784, paper | 0.929 / 0.453 | 0.938 / 0.466 | 0.860 / 0.294 | 0.909 / 0.310 |
| 784, here | 0.9272 / 0.4566 | 0.9406 / 0.4623 | 0.8583 / 0.2881 | 0.9031 / 0.3116 |

Set `--memory_size`. The "no SCL" column is `--epochs_num 0`.

### Table 7, the feature-layer ablation

Set `UCAD_FEATURE_BLOCK`. One seed per cell.

| block | MVTec paper | MVTec here | VisA paper | VisA here |
|---|---|---|---|---|
| 1 | 0.840 / 0.399 | 0.9200 / 0.4495 | 0.806 / 0.143 | 0.8272 / 0.1747 |
| 3 | 0.934 / 0.451 | 0.9323 / 0.4552 | 0.876 / 0.283 | 0.8737 / 0.2826 |
| 5 | 0.930 / 0.456 | 0.9259 / 0.4512 | 0.874 / 0.300 | 0.8638 / 0.2982 |
| 7 | 0.936 / 0.444 | 0.9311 / 0.4399 | 0.872 / 0.267 | 0.8692 / 0.2700 |
| 9 | 0.906 / 0.420 | 0.9292 / 0.4334 | 0.853 / 0.248 | 0.8611 / 0.2538 |

Blocks 3 to 9 land within 0.011 of the published cell on VisA and within 0.005 on MVTec, apart from
block 9 on MVTec at +0.023. Block 1 does not: +0.080 image AUROC on MVTec and +0.051 pixel AUPR.
The shape matches - the first block is much the worst, the middle blocks are flat within about 0.01,
and quality falls off again by block 9 - so the paper's choice to keep block 5 "for simplicity" is
well supported: the best block here beats it by 0.006 on MVTec and 0.010 on VisA, both at one seed
and both smaller than the spread between seeds elsewhere in this document.

### Forgetting

The Forgetting Measure needs the matrix of "concept j's score after concept k was learned", for every
k >= j. The released code never revisits a concept, so it cannot produce that matrix;
`UCAD_LOG_FM=1` does, by routing and re-scoring every concept learned so far each time a new one
finishes. It costs O(T^2) test passes on top of training.

| | tasks | misrouted images | avg FM (Eq. 7) | worst single drop | paper |
|---|---|---|---|---|---|
| MVTec, 0 epochs | 15 | 0 | 0.000000 | 0.000000 | - |
| MVTec, 25 epochs | 15 | 0 | 0.000000 | 0.000000 | 0.010 |
| VisA, 0 epochs | 12 | 0 | 0.000000 | 0.000000 | - |
| VisA, 25 epochs | 12 | 0 | 0.000000 | 0.000000 | 0.039 |

All four are complete runs over every concept of the benchmark.

Not approximately zero - exactly. Every earlier concept repeats its just-learned AUROC bit for bit as
later concepts join the memory:

```
FM_MATRIX learned:0 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:1 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:1 eval:1 routed:1 auroc:0.715704647676162
FM_MATRIX learned:2 eval:0 routed:0 auroc:0.9984126984126984
FM_MATRIX learned:2 eval:1 routed:1 auroc:0.715704647676162
```

That follows from the architecture: nothing is shared between concepts, so the only thing that could
move an earlier concept's score is the key routing to a different concept, and `routed` equals `eval`
on every one of the 1725 MVTec and 2162 VisA test images. The published 0.010 and 0.039 therefore have
no source in this code or in this design; the honest number for this architecture is 0.

## Things that will trip you up

**The task-agnostic inference phase never runs.** In `run_ucad.py` the whole phase that routes a test
image by its key, retrieves that concept's prompt and knowledge, and evaluates every concept once all
of them are learned sits inside a triple-quoted string between `# Inference` and the results writing.
`results.csv` comes from the training loop instead: each concept evaluated immediately after it is
learned, against its own bank, with its identity known by construction. `UCAD_INFERENCE=1` runs the
phase. Untrained it changes nothing, because the routing is exact.

**Three sizes are hardcoded** and will break any change of resolution: the SAM label map at 14x14 in
`forward_head`, the anomaly map at 224x224 in `run_ucad.py`, and a k-means prototype reshape at
`196*4*768` which is dead on the path the code takes but would break for a batch size that is not a
multiple of four. This branch derives the first two from the model and the dataset.

**`args_dict.npy` is misleading.** It carries `length=5` and `batch_size=24`, neither of which
reaches the model, and `dataset='Split-CIFAR100'` - it is a leftover from the DualPrompt codebase the
prompt machinery was taken from. What it does supply is `opt='adam'` and `sched='constant'`, so no
learning-rate schedule ever runs.

**The SAM label map is resized bilinearly.** `cv2.resize` without an interpolation argument averages
the segment ids the map holds, and the loss compares those ids with `==`, so cells on segment
boundaries match nothing. `UCAD_SAM_INTERP=nearest` samples them instead.
