#!/bin/bash

#SBATCH --account=nlp                 # Specify the account
##SBATCH --partition=sphinx               # Specify the partition
##SBATCH --nodelist=sphinx9      # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=320GB                         # Memory request
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=logs/run_nlp_%j.out
#SBATCH --error=logs/run_nlp_%j.err

#SBATCH --job-name=dclm_process
#SBATCH --partition=sc-loprio
#SBATCH --constraint=(80G|141G)&hopper
##SBATCH --constraint=[ampere|hopper]
#SBATCH --exclude=tiger6,pasteur1,tiger-hgx-1

set -euo pipefail

source /nlp/scr/kathli/miniconda3/etc/profile.d/conda.sh
conda activate cyagent

export MAX_PROMPT_LENGTH=${3:-3072}
export MAX_RESPONSE_LENGTH=${4:-384}

TASK_NAME=${1:-"dynastic"}

TASK_DIR=${2:-"benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic"}

CYAGENT_LOG_DIR="logs/cyagent/${TASK_NAME}"
export LOG_DIR="cybench/$CYAGENT_LOG_DIR"
export CONTAINER_NAME="cybench_${TASK_NAME}"

# if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
#     rm -rf "$LOG_DIR"
# fi
# mkdir -p "$LOG_DIR"

# -------------- CLEANUP ----------------
# Trap SIGINT (Ctrl+C) and SIGTERM (kill), to cleanup
cleanup() {
    echo "[$(date)] Caught interrupt signal, killing all child processes..."
    # Kill train process
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill "$TRAIN_PID"
    fi
    # Kill worker processes
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
    exit 1
}

trap cleanup SIGINT SIGTERM

# -------------- VERL ----------------
echo "[$(date)] Launching verl training job..."

cd /nlp/scr/kathli/repos/TinyZero

export N_GPUS=4
# export BASE_MODEL=model/Qwen2.5-0.5B-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
export BASE_MODEL=model/Qwen2.5-3B-instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
# export BASE_MODEL=model/Qwen3-0.6B/snapshots/6130ef31402718485ca4d80a6234f70d9a4cf362
# export BASE_MODEL=model/Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
export DATA_DIR=data/cybench_512
export ROLLOUT_TP_SIZE=1
# export EXPERIMENT_NAME=$DATA_DIR-qwen2.5-0.5B-instruct
export EXPERIMENT_NAME=$DATA_DIR-qwen2.5-3B-instruct-$TASK_NAME
# export EXPERIMENT_NAME=$DATA_DIR-qwen3-0.6b
# export EXPERIMENT_NAME=$DATA_DIR-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./wandb_login.sh
bash ./scripts/train_cyagent.sh 2>&1 | tee "verl_out_${TASK_NAME}.log" &
TRAIN_PID=$!

# -------------- CYBENCH ----------------
NUM_CYBENCH_INSTANCES=1
CYBENCH_DIR="/nlp/scr/kathli/repos/TinyZero/cybench"

echo "[$(date)] Starting $NUM_CYBENCH_INSTANCES CPU workers..."
WORKER_PIDS=()
for i in $(seq 1 $NUM_CYBENCH_INSTANCES); do
    (
        cd "$CYBENCH_DIR"
        CUDA_VISIBLE_DEVICES="" ./run_train_interface.sh --task_dir "${TASK_DIR}" --max_iterations 5 --max_input_tokens $MAX_PROMPT_LENGTH --max_output_tokens $MAX_RESPONSE_LENGTH --model Qwen-train/Qwen2.5-3B-instruct --iterations_until_hint 1 --responses_to_keep 6 --observations_to_keep 6 --task_objective "answer a task" --verl_log_dir "${CYAGENT_LOG_DIR}" > "${SLURM_SUBMIT_DIR}/task_${TASK_NAME}_worker_${i}.log" 2>&1
    ) &
    WORKER_PIDS+=($!)
done

# Wait for training to finish
wait $TRAIN_PID
echo "[$(date)] Training complete. Killing workers..."

# Kill all CPU workers
for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "Killing worker PID $pid"
        kill "$pid"
    fi
done

# Optional: wait to make sure they cleanly exit
wait