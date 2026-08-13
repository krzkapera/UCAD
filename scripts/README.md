# scripts

Everything needed to run this repository's experiments, kept out of the model code so that
`run_ucad.py` and `patchcore/` stay as close to the published version as possible.

| file | what it is for |
|---|---|
| `launch.py` | entry point; restores the NumPy 1.x aliases the code still uses and caps the process's address space, then runs `run_ucad.py` |
| `run_benchmark.sbatch` | one benchmark per submission, configured through the environment |
| `visa_sam_b_masks.py` | rewrites VisA SAM label maps into the 8-bit, image-named form `run_ucad.py` reads |

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
prefix untrained.

`UCAD_MVTEC_MASKS` and `UCAD_VISA_MASKS` point the SAM masks at a different directory than the
`mvtec2d-sam-b` and `visa-sam-b` the code otherwise assumes.
