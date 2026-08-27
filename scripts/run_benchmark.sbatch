#!/bin/bash -l
#SBATCH --job-name=ucad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# One benchmark per submission, configured through the environment. Anything the model code reads
# (UCAD_VIT, UCAD_VIT_WEIGHTS, UCAD_FEATURE_BLOCK, UCAD_PROMPT_LEN, UCAD_LOSS, UCAD_NO_PROMPT,
# UCAD_NORMALIZE_FEATURES, UCAD_SAM_INTERP, UCAD_DUMP_SCORES, ...) is passed through by
# --export=ALL and defaults to the released behaviour; see scripts/README.md. What this file adds:
#
#   UCAD_DATASET        visa | mvtec                     (default visa)
#   UCAD_SEED           seed passed to --seed            (default 0)
#   UCAD_EPOCHS         epochs per concept               (default 25)
#   UCAD_CATEGORIES     space separated subset           (default: all of them)
#   UCAD_DATA_ROOT      where the images live            (required)
#   UCAD_VENV           python environment to use        (required)
#   UCAD_RUN_DIR        where results and the log go     (default runs/<dataset>_<seed>_<jobid>)
#   UCAD_LOG_PROJECT    results subdirectory             (default REFEXP, which scripts/collect.py reads)
#   UCAD_MEMORY         --memory_size, the bank          (default 196)
#   UCAD_RESIZE         --resize and --imagesize         (default 224)
#   UCAD_NN             --anomaly_scorer_num_nn          (default 1; inert, see MEASUREMENTS.md)
#   UCAD_PATCHSIZE      --patchsize                      (default 1; 3 collapses the method)
#   UCAD_CKPT_DIR       resume point; a re-submission with the same directory continues
#   UCAD_LOG_EPOCHS     print image AUROC and pixel AP after every epoch
#   UCAD_EVAL_UNTRAINED evaluate once before the first epoch
#
# Account and partition are site specific, so pass them on the command line:
#   sbatch -A <grant> -p <partition> --export=ALL,UCAD_DATASET=visa,... scripts/run_benchmark.sbatch

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

DATASET=${UCAD_DATASET:-visa}
SEED=${UCAD_SEED:-0}
EPOCHS=${UCAD_EPOCHS:-25}
RESIZE=${UCAD_RESIZE:-224}
MEMORY=${UCAD_MEMORY:-196}
RUN_DIR=${UCAD_RUN_DIR:-$ROOT/runs/${DATASET}_${SEED}_${SLURM_JOB_ID:-manual}}
mkdir -p "$RUN_DIR"

if [ "$DATASET" = "mvtec" ]; then
    DEFAULT_CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
else
    DEFAULT_CATEGORIES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
fi

D_FLAGS=""
for category in ${UCAD_CATEGORIES:-$DEFAULT_CATEGORIES}; do D_FLAGS="$D_FLAGS -d $category"; done

echo "UCAD dataset=$DATASET seed=$SEED epochs=$EPOCHS resize=$RESIZE memory=$MEMORY" \
     "weights=${UCAD_VIT_WEIGHTS:-orig_in21k} vit=${UCAD_VIT:-default} block=${UCAD_FEATURE_BLOCK:-5}" \
     "prompt_len=${UCAD_PROMPT_LEN:-1} loss=${UCAD_LOSS:-code} no_prompt=${UCAD_NO_PROMPT:-0}" \
     "normalize=${UCAD_NORMALIZE_FEATURES:-0} nn=${UCAD_NN:-1} patchsize=${UCAD_PATCHSIZE:-1}"

"${UCAD_VENV}/bin/python" scripts/launch.py --gpu 0 --seed "$SEED" --memory_size "$MEMORY" --epochs_num "$EPOCHS" \
    --log_group FULL --log_project "${UCAD_LOG_PROJECT:-REFEXP}" "$RUN_DIR/results" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn "${UCAD_NN:-1}" --patchsize "${UCAD_PATCHSIZE:-1}" \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize "$RESIZE" --imagesize "$RESIZE" $D_FLAGS --num_workers 2 mvtec "${UCAD_DATA_ROOT}" \
    2>&1 | tee "$RUN_DIR/run.log" |
    grep --line-buffered -E "SINGLE_EPOCH|GEOMETRY|PROMPT |FM_MATRIX|Evaluating dataset|Traceback|CUDA error|MemoryError"

echo "===DONE=== exit=${PIPESTATUS[0]} results=$RUN_DIR"
