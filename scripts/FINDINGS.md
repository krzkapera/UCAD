# What this code does, measured

Every number below comes from this repository, run on Helios. Image AUROC and pixel AUPR, averaged
over the benchmark's categories, in that order.

The configuration is the one the paper describes and this code implements: ViT-B/16 pretrained on
ImageNet-21k, features after block 5, prompt length 1, a bank of 196 vectors per concept, 224px,
batch 8, 25 epochs, VisA on its official `split_csv/1cls.csv` split. Where a number rests on fewer
than three seeds it says so.

## The paper reproduces

Headline tables:

| | paper | here |
|---|---|---|
| MVTec, image AUROC | 0.930 | 0.9259 (3 seeds) |
| MVTec, pixel AUPR | 0.456 | 0.4512 (3 seeds) |
| VisA, image AUROC | 0.874 | 0.8638 |
| VisA, pixel AUPR | 0.300 | 0.2982 |

Table 5, the module ablation:

| | MVTec paper | MVTec here | VisA paper | VisA here |
|---|---|---|---|---|
| no CPM, no SCL | 0.693 / 0.183 | 0.6692 / 0.1621 | 0.584 / 0.050 | 0.5862 / 0.0491 |
| CPM, no SCL | 0.894 / 0.426 | 0.9153 / 0.4255 | 0.786 / 0.251 | 0.7872 / 0.2455 |
| CPM and SCL | 0.930 / 0.456 | 0.9259 / 0.4512 | 0.874 / 0.300 | 0.8638 / 0.2982 |

Table 6, the knowledge-size ablation:

| bank | MVTec no SCL | MVTec SCL | VisA no SCL | VisA SCL |
|---|---|---|---|---|
| 196, paper | 0.894 / 0.426 | 0.930 / 0.456 | 0.786 / 0.251 | 0.874 / 0.300 |
| 196, here | 0.9153 / 0.4255 | 0.9259 / 0.4512 | 0.7872 / 0.2455 | 0.8638 / 0.2982 |
| 392, paper | 0.921 / 0.452 | 0.936 / 0.461 | 0.818 / 0.255 | 0.893 / 0.307 |
| 392, here | 0.9203 / 0.4485 | 0.9401 / 0.4614 | 0.8315 / 0.2716 | 0.8852 / 0.3069 |
| 784, paper | 0.929 / 0.453 | 0.938 / 0.466 | 0.860 / 0.294 | 0.909 / 0.310 |
| 784, here | 0.9272 / 0.4566 | 0.9406 / 0.4623 | 0.8583 / 0.2881 | 0.9031 / 0.3116 |

Most cells land within 0.01 of the published one; the worst is MVTec's 196-vector no-SCL cell, at
+0.021. Getting there needs three things that are easy to get wrong: the ImageNet-21k checkpoint
rather than any of timm's ImageNet-1k fine-tunes, which is worth up to 0.07 on VisA; VisA's official
split rather than a per-category folder copy, worth a further 0.06 in the other direction; and
block 5.

## What the contrastive loss contributes: nothing

Run the same 25 epochs with the loss forced to zero. The prompt never moves, so every epoch scores
with the same model, and the only thing that differs between epochs is the coreset draw.

| | MVTec | VisA |
|---|---|---|
| untrained, one reading | 0.9153 / 0.4255 | 0.7872 / 0.2455 |
| **zero loss, 25 epochs, reported the way this code reports** | **0.9271 / 0.4520** | **0.8669 / 0.3009** |
| SCL, 25 epochs, same reporting | 0.9259 / 0.4512 | 0.8638 / 0.2982 |

Four comparisons out of four, learning nothing scores at least as well as learning with SCL. The
+0.088 image AUROC that Table 5 credits to SCL on VisA is +0.080 here without a single gradient step.

The mechanism is in how the code reports. It scores the test set after every epoch, rescales each
epoch's scores to 0..1, averages every epoch so far, and keeps the epoch whose image AUROC on the
test set is highest. Averaging cancels independent noise; a maximum over 25 noisy readings of one
test set is biased upward. Both need the epochs to differ from each other. In the no-SCL row there is
no loss, hence no training, hence 25 identical models: averaging 25 copies is the identity, and the
maximum of 25 equal numbers is that number. Both mechanisms are off in one row of the ablation and on
in the other.

So the ablation does not compare a loss against no loss. It compares one reading against the selected
mean of 25, and SCL's role is to supply the variation that the reporting exploits. Coreset randomness
supplies just as much.

A single model's own image AUROC on VisA starts at 0.787 untrained, peaks near 0.80 at epoch 2 or 3 -
which epoch varies with the seed - and falls to 0.74 by epoch 25, still falling at 100. The loss has
no equilibrium: `-cos` on same-segment pairs is minimised when a segment collapses to a point,
`exp(cos)` on different-segment pairs when segments are maximally spread, and nothing anchors the
features to where they started. Its optimum is a degenerate embedding. A defect sits inside a
segment, so collapsing segments removes exactly the variation the nearest-neighbour score reads.

Writing the loss as the paper's Eq. 3 writes it - a plain difference of cosines, no temperature and
no exponential - recovers 0.005 on VisA and leaves the shape unchanged: up for two or three epochs,
down thereafter.

## What CPM contributes: a great deal

The first ablation row is the honest one, because neither side of it trains. Replacing the
per-concept key-prompt-knowledge memory with a single bank that each task overwrites costs 0.25 image
AUROC on MVTec and 0.20 on VisA. That is the paper's real result.

Routing is exact. With every concept in memory, the key sends every test image to its own concept on
both benchmarks: the routed reading and the each-concept-against-its-own-bank reading agree to four
decimals, at 0.9153 / 0.4255 and 0.7872 / 0.2455.

## The evaluation the released code never runs

That routed reading needed a change to obtain. In `run_ucad.py` the whole task-agnostic inference
phase - route by key, retrieve that concept's prompt and knowledge, evaluate every concept once all
of them have been learned - sits between `# Inference` and the results writing **inside a
triple-quoted string**, and never executes. `results.csv` is written from the training loop instead,
where each concept is evaluated immediately after it is learned, against its own bank.

Three things follow. The key routing, which is what makes the method task-agnostic, is not exercised
by a run of this code. No concept is re-evaluated after later concepts are learned, so nothing in a
run can measure forgetting, and the published FM values have no source in this code path. And every
number the code produces knows the task identity by construction, which is the assumption the paper
sets out to remove.

`UCAD_INFERENCE=1` runs the phase. Untrained, it changes none of the averages, because routing is
perfect - but that is now a measurement rather than something a reader had to assume.

After 25 epochs the phase reads lower than `results.csv` does, and the reason is not forgetting.
What it stores per concept is one prompt and one bank, so it scores each concept with a single
model, where `results.csv` reports the average of 25 epochs. Read that way - task-agnostic, single
model, after the whole sequence has been learned - the method scores 0.9189 / 0.4248 on MVTec and
0.7801 / 0.2337 on VisA, against 0.9153 / 0.4255 and 0.7872 / 0.2455 for the same thing untrained.
That is the most honest number this code can produce, and training moves it by +0.004 and -0.007.

## Smaller things

**The label map is resized bilinearly.** `cv2.resize` without an interpolation argument averages the
SAM segment ids the map holds, and the loss compares those ids with `==`, so cells on segment
boundaries match nothing. Sampling them instead is worth +0.008 image AUROC over 25 epochs on five
MVTec categories.

**The prompt in the paper is not the prompt in the code.** The paper adds a prompt to each layer's
input, `k^i = f^i(k^{i-1} + p^i)`, and accounts for it as (15, 7, 768) floats, which the stated
23.28MB total confirms. The code does prefix tuning on twelve layers with separate keys and values,
24x768 per task, inherited with the rest of the prompt machinery from DualPrompt - `args_dict.npy`
still carries `dataset='Split-CIFAR100'` from that codebase.

**Two numbers in the paper disagree.** The text gives a learning rate of 0.0005, the appendix table
gives 0.00005 for this method. The code uses 0.0005, with `sched='constant'`, so no schedule runs.

**The backbone flags in the README command do nothing.** `-b wideresnet50 -le layer2 -le layer3` is
carried over from PatchCore's script; `PatchCore.load` builds a ViT unconditionally.

**Three sizes were hardcoded**: the SAM label map at 14x14, the anomaly map at 224x224, and a k-means
prototype reshape at `196*4*768` that is dead on the path the code takes and would break for any
batch size that is not a multiple of four.

**The block and the bank were both explored by the paper** (Tables 7 and 6) and both matter more than
the loss. Block 7 or 9 is worth up to +0.02 over block 5 at no cost; the paper kept 5 "for
simplicity". A bank of 784 is worth +0.07 on VisA untrained - at four times the memory, which is the
one thing the method is trying to economise.
