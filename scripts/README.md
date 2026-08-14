# scripts

Everything needed to run this repository's experiments, kept out of the model code so that
`run_ucad.py` and `patchcore/` stay as close to the published version as possible.

| file | what it is for |
|---|---|
| `launch.py` | entry point; restores the NumPy 1.x aliases the code still uses and caps the process's address space, then runs `run_ucad.py` |
| `run_benchmark.sbatch` | one benchmark per submission, configured through the environment |
| `visa_sam_b_masks.py` | rewrites VisA SAM label maps into the 8-bit, image-named form `run_ucad.py` reads |
| `FINDINGS.md` | what the measurements say: which checkpoint reproduces the paper, what the loss is worth, and which unvaried settings are worth more |

## Running a benchmark

```bash
sbatch -A <grant> -p <partition> \
  --export=ALL,UCAD_DATASET=visa,UCAD_SEED=0,UCAD_DATA_ROOT=/path/to/visa,UCAD_VENV=/path/to/.venv \
  scripts/run_benchmark.sbatch
```

## The instrumentation the model code carries

Three environment variables are read inside `run_ucad.py`. With none of them set the run behaves
exactly as the published code does.

`UCAD_CKPT_DIR` saves each finished concept and resumes from it, so a job that runs out of wall time
can be re-submitted and will continue from the concept it reached.

`UCAD_LOG_EPOCHS` prints one line per epoch with that epoch's own image AUROC and pixel AP, before
the epochs are averaged together:

```
SINGLE_EPOCH category:0 name:mvtec_candle epoch:7 auroc:0.5137 pixel_ap:0.0163
```

The training loop otherwise only reports the running average over every epoch so far, and keeps the
epoch whose average scores best on the test set, so a single epoch's own reading is not recoverable
from the normal output.

`UCAD_EVAL_UNTRAINED` runs one evaluation before the first epoch, which measures the model with its
prefix untrained. That evaluation builds a memory bank, and the coreset's random projection draws
from the global RNG, so the epochs that follow it see a different random stream than they would
have. The untrained reading itself is unaffected, but do not compare a trained epoch from a run with
this flag against one from a run without it. To measure only the untrained model, pass
`--epochs_num 0`: the loop then runs the evaluation and nothing else.

`UCAD_MVTEC_MASKS` and `UCAD_VISA_MASKS` point the SAM masks at a different directory than the
`mvtec2d-sam-b` and `visa-sam-b` the code otherwise assumes.

## Varying the backbone, the grid and the loss input

Four more variables exist so that the choices baked into `patchcore/` can be varied without editing
it. All four default to what the released code does.

`UCAD_VIT_WEIGHTS` selects which pretrained checkpoint is loaded, as a Hugging Face tag. This one
deserves attention: `default_cfgs` in `patchcore/vision_transformer.py` has the entry for
`vit_base_patch16_224` edited away from timm's augreg checkpoint - that line is still there,
commented out - and pointed at `imagenet21k/ViT-B_16.npz`, which is the original ImageNet-21k
release with no ImageNet-1k fine-tuning. The default here is that checkpoint, `orig_in21k`. The
docstring above the model function still describes the augreg one.

`UCAD_VIT` names the timm ViT to build. The patch size sets the feature grid: `vit_base_patch32_224`
gives 7x7, the default `vit_base_patch16_224` gives 14x14, `vit_base_patch8_224` gives 28x28, and
`vit_base_patch16_384` gives 24x24 with the patch size unchanged, which separates a finer grid from
simply having more tokens. Pass `--resize 384 --imagesize 384` for the 384 variants.

`UCAD_FEATURE_BLOCK` is the transformer block the patch features are read after, 5 by default.

`UCAD_PROMPT_LEN` is the number of prefix tokens per layer, 1 by default. The paper reports a prompt
of shape (15, 7, 768) and `args_dict.npy` carries `length=5`; neither reaches the model.

`UCAD_LOG_GEOMETRY` prints, after every epoch, the mean cosine of patch pairs inside one SAM segment
and across two:

```
GEOMETRY category:0 name:visa_candle epoch:7 within:0.83 between:0.21
```

The loss drives the first towards 1 and the second towards -1, and its optimum is the degenerate
embedding where that is reached. Reading them beside `SINGLE_EPOCH` shows how far a run has gone
towards it and what that costs.

`UCAD_NO_CPM=1` removes the key-prompt-knowledge memory from the final evaluation: no routing, and
every task scored against the last task's knowledge, which is the "single Knowledge base, reset every
time a new task was introduced" of the paper's first ablation row. Pass `--epochs_num 0` with it for
the row that has neither CPM nor SCL.

`UCAD_SAM_INTERP=nearest` samples the SAM label map when resizing it to the feature grid instead of
averaging it. The map holds segment ids and the loss compares them for equality, so bilinear - which
is `cv2.resize`'s default and therefore what runs otherwise - leaves cells on segment boundaries
with averaged values that equal no id at all.
