#!/bin/bash -l
#SBATCH --job-name=ucad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# One benchmark per submission, configured through the environment:
#
#   UCAD_DATASET        visa | mvtec                     (default visa)
#   UCAD_SEED           seed passed to --seed            (default 0)
#   UCAD_EPOCHS         epochs per concept               (default 25)
#   UCAD_CATEGORIES     space separated subset           (default: all of them)
#   UCAD_DATA_ROOT      where the images live            (required)
#   UCAD_VENV           python environment to use        (required)
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
RUN_DIR=${UCAD_RUN_DIR:-$ROOT/runs/${DATASET}_${SEED}_${SLURM_JOB_ID:-manual}}
mkdir -p "$RUN_DIR"

if [ "$DATASET" = "visa" ]; then
    DEFAULT_CATEGORIES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
else
    DEFAULT_CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
fi

D_FLAGS=""
for category in ${UCAD_CATEGORIES:-$DEFAULT_CATEGORIES}; do D_FLAGS="$D_FLAGS -d $category"; done

"${UCAD_VENV}/bin/python" scripts/launch.py --gpu 0 --seed "$SEED" --memory_size 196 --epochs_num "$EPOCHS" \
    --log_group FULL --log_project UCAD "$RUN_DIR/results" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 $D_FLAGS --num_workers 2 mvtec "${UCAD_DATA_ROOT}" \
    2>&1 | tee "$RUN_DIR/run.log" |
    grep --line-buffered -E "SINGLE_EPOCH|Evaluating dataset|Traceback|CUDA error|MemoryError"

echo "===DONE=== exit=${PIPESTATUS[0]} results=$RUN_DIR"
