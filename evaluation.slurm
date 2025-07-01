#!/bin/bash
#SBATCH --job-name=smollm3-multi-if
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00


# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.4

# Refresh Weka on h4 cache
echo "Refreshing Weka filesystem..."
find -L /fsx/h4/ -type f | xargs -d '\n' -r -n512 -P64 weka fs tier fetch

# Needed for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -x -e

# Record start time for timing the evaluation
SCRIPT_START_TIME=$(date +%s)
echo "Evaluation started at: $(date)"

source ~/.bashrc
source /fsx/lewis/git/hf/trl-internal/trl-internal/bin/activate

TASK_NAME="multi_if"

MODEL_ID=$1
MODEL_REVISION=$2
# Optional args
[ -z "$3"] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$3
[ -z "$4"] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$4
# $5 is reserved for system_prompt, see line 65

LM_EVAL_REPO_ID="HuggingFaceTB/smollm3-post-training-leaderboard"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//__/g') # replaces / with __
DETAILS_REPO_ID="HuggingFaceTB/details_$MODEL_NAME"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME/$TIMESTAMP"
# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false

NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Running VLLM script ..."
echo "Eval results will be saved to $OUTPUT_DIR"
# Env var needed to handle https://github.com/vllm-project/vllm/issues/6152
VLLM_WORKER_MULTIPROC_METHOD=spawn python multi_turn_instruct_following_eval_vllm.py  --model_path $MODEL_ID --revision $MODEL_REVISION \
    --languages Spanish French Italian Portuguese  \
    --tensor_parallel_size $NUM_GPUS \
    --output_dir $OUTPUT_DIR
# VLLM_WORKER_MULTIPROC_METHOD=spawn python examples/smollm3/scripts/alpaca_eval/generate_vllm.py --model_name_or_path $MODEL_ID --model_revision $MODEL_REVISION \
#     --tp $NUM_GPUS \
#     --trust_remote_code $TRUST_REMOTE_CODE \
#     --dataset_name tatsu-lab/alpaca_eval \
#     --dataset_split eval \
#     --prompt_column instruction \
#     --max_new_tokens 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --output_dir $OUTPUT_DIR \
#     ${5:+--system-prompt "$(echo "$5" | base64 --decode)"}

OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/ -type f -name "*result*.json")
for filepath in $OUTPUT_FILEPATHS; do
    # Add system prompt to results (empty if not provided)
    if [ -n "$5" ]; then
        SYSTEM_PROMPT=$(echo "$5" | base64 --decode)
    else
        SYSTEM_PROMPT=""
    fi
    jq --arg prompt "$SYSTEM_PROMPT" '.config_general.system_prompt = $prompt' "$filepath" > "${filepath}.tmp" && mv "${filepath}.tmp" "$filepath"
    
    echo "Uploading $filepath to Hugging Face Hub..."
    filename=$(basename -- "$filepath")
    for attempt in {1..20}; do
        if huggingface-cli upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename; then
            echo "Upload succeeded for $filepath"
            break
        else
            echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in 5 seconds..."
            sleep 5
        fi
    done
done

# TODO: add details upload
# echo "Uploading details to Hugging Face Hub..."
# DETAILS_FILEPATHS=$(find $OUTPUT_DIR/weighted_alpaca_eval_gpt4_turbo/details/ -type f \( -name "*.parquet" \))
# echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
# python examples/smollm3/scripts/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --revision $MODEL_REVISION.$TASK_NAME.$TIMESTAMP
    
echo "Cleaning up ..."
rm -rf $OUTPUT_DIR

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Evaluation completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes (${TOTAL_SECONDS} seconds)"

echo "Done!"