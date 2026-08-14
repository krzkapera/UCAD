# What this code does, measured

Every number below comes from this repository, run on Helios through `scripts/run_benchmark.sbatch`.
Image AUROC and pixel AUPR, averaged over the benchmark's categories. Two readings appear
throughout, because the code writes both and only ever reports the first:

- **bank 196** - `results/`, the memory the paper describes, one image's worth of patches per concept
- **bank 1960** - `results_nolimit/`, computed in the same run from `--basic_size` and never mentioned

"0 epochs" is a single untrained model, no ensembling and no epoch selection. "25 epochs" is what the
code reports: the mean of the 25 per-epoch score sets, at the epoch whose image AUROC on the test set
is highest.

The dataset is the per-category folder copy this code requires, not VisA's official `1cls.csv`
split, so these numbers are internally comparable but not comparable to the wider literature.

## The configured checkpoint is not the one that reproduces the paper

`default_cfgs` in `patchcore/vision_transformer.py` has the `vit_base_patch16_224` entry edited: the
augreg line is commented out and `imagenet21k/ViT-B_16.npz` put in its place, which is the original
ImageNet-21k release with no ImageNet-1k fine-tuning. The docstring above the model function still
describes the augreg checkpoint. With `timm==0.6.7` as `environment.yaml` pins, `register_model`
honours that edit, so a run of this repository as released loads the 21k weights.

VisA, twelve categories, image AUROC, three seeds:

| checkpoint | 25 epochs, bank 196 | 0 epochs, bank 196 |
|---|---|---|
| `orig_in21k` - the active line | **0.8069 +- 0.0126** | 0.7733 |
| `augreg_in21k_ft_in1k` - the commented-out line | **0.8634 +- 0.0150** | 0.7937 |
| `augreg2_in21k_ft_in1k` - what current timm resolves | **0.8725 +- 0.0046** | 0.7703 |
| the paper | 0.874 | |

The checkpoint the code disables lands on the published figure. The checkpoint it enables is 0.067
below it, five standard deviations away. MVTec, one seed, shows the same ordering: 0.9254 for
`orig_in21k`, 0.9353 for `augreg`, 0.9435 for `augreg2`, against 0.930 published.

## The contrastive loss makes the model worse, on every checkpoint and every configuration

`UCAD_LOG_EPOCHS` reports each epoch's own image AUROC before the epochs are averaged. VisA, twelve
categories, single model, bank 196:

| configuration | untrained | after 1 epoch | after 25 epochs | best epoch |
|---|---|---|---|---|
| released, `augreg`, 3 seeds | 0.7937 | 0.759 / 0.790 / 0.791 | 0.700 / 0.693 / 0.679 | 2, 2, 6 |
| released, `orig_in21k`, 3 seeds | 0.7733 | 0.742 / 0.732 / 0.737 | 0.734 / 0.700 / 0.708 | 4, 10, 5 |
| block 7, bank 784 | 0.8535 | 0.8545 | 0.8101 | 5 (0.8750) |

Training costs 0.09 to 0.11 image AUROC over twenty-five epochs at the released settings, and 0.044
at the strongest configuration found here. What the reported number recovers is the epoch ensemble
and the test-set epoch choice, not the loss. The epoch that scores best is an early one, and which
one it is moves between seeds.

## Two settings that were never varied are worth more than the training

### Memory

The bank size is the single largest lever. Both benchmarks, 0 epochs, block 7:

| bank | VisA, 14x14 | MVTec, 28x28 |
|---|---|---|
| 196 | 0.8122 | - |
| 392 | 0.8198 | 0.9524 |
| 784 | 0.8535 | 0.9639 |
| 1568 | 0.8673 | 0.9690 |
| 1960 | 0.8802 | - |
| 3136 | 0.8826 | 0.9716 |

Still rising at 3136. The 1960 column of `results_nolimit` is in every run this code has ever
produced: **on VisA an untrained model with that bank already scores 0.8876, above the 0.874 the
paper reports for the trained one.**

This is not free - it is four to sixteen times the memory the method advertises, and fixed small
memory is the method's claim. It does say that the published figures measure a memory budget as much
as they measure a loss.

### Feature block

Features are read after block 5 of 12. Nothing in the paper argues for 5. Full benchmarks, 0 epochs,
bank 196:

| block | MVTec | VisA |
|---|---|---|
| 5, as released | 0.9177 | 0.7937 |
| 7 | - | 0.8122 |
| 9 | 0.9395 | 0.7795 |

The best block differs by benchmark, so this is a tuning knob rather than a fix - but block 5 is not
the right value for either benchmark, and on MVTec the untrained model at block 9 scores 0.9395,
above the 0.930 published for the trained one.

## Grid resolution helps only if the bank grows with it

The grid was hardcoded at 14x14 in the loss and in two reshapes, so this could not be varied before.
MVTec, five categories, 0 epochs:

| grid | bank 196 | bank = tokens per image |
|---|---|---|
| 7x7 (patch32) | 0.7659 | - |
| 14x14 (patch16) | 0.8272 | 0.8481 |
| 24x24 (patch16, 384px) | 0.8154 | 0.8367 |
| 28x28 (patch8) | 0.7575 | **0.8633** |

At a fixed 196 vectors a finer grid *loses*, because the coreset compresses four times harder. Give
the bank room and the ordering reverses. At equal bank size the finer grid wins on MVTec (full
benchmark, bank 784, block 7: 0.9639 against 0.9577, and pixel AUPR 0.5312 against 0.4717) and loses
on VisA (0.8321 against 0.8535), which fits MVTec's defects being smaller than one patch of the
coarse grid and VisA's being larger and more contextual.

## The best untrained configurations beat the published numbers

| | configuration | image AUROC | pixel AUPR | published |
|---|---|---|---|---|
| MVTec | block 7, 28x28, bank 784, 0 epochs | **0.9639** | **0.5312** | 0.930 |
| VisA | block 7, 14x14, bank 3136, 0 epochs | **0.8826** | 0.3299 | 0.874 |
| VisA | block 7, 14x14, bank 1960, 0 epochs | 0.8802 | 0.3300 | 0.874 |

At the paper's own memory budget of 196 the untrained model reaches 0.9395 on MVTec, still above the
published figure, and 0.8122 on VisA, below it.

## Smaller things

**The label map is resized bilinearly.** `cv2.resize` without an interpolation argument averages the
SAM segment ids the map holds, and the loss compares those ids with `==`, so cells on segment
boundaries end up matching nothing. Sampling them instead (`UCAD_SAM_INTERP=nearest`) is worth
+0.008 image AUROC and -0.006 pixel AUPR over 25 epochs on five MVTec categories - real but not the
explanation for anything.

**The prompt length in the paper is not the one in the code.** The paper reports a prompt of shape
(15, 7, 768) and `args_dict.npy` carries `length=5`; the model is built with `prompt_length=1`. Over
25 epochs on five MVTec categories: length 1 gives 0.8396, length 5 gives 0.8562, length 7 gives
0.8463. The direction matches the paper, the magnitude does not.

**The backbone flags in the README command do nothing.** `-b wideresnet50 -le layer2 -le layer3` is
carried over from PatchCore's script; `PatchCore.load` builds a ViT unconditionally and the
wideresnet is never used.

**Three sizes were hardcoded.** The SAM label map at 14x14, the anomaly map at 224x224, and the
k-means prototype reshape at `196*4*768` - the last of which is dead code on the path
`run_ucad.py` actually takes, and would break for any batch size that is not a multiple of four.
