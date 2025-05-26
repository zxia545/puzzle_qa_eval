#!/bin/bash
source activate vllm


# Configuration for generation models
declare -a model_paths=(
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
)

declare -a model_names=(
    # "qwen2.5-7b"
    # "qwen2.5-32b-chat"
    # "qwen2.5-14b-chat"
    "qwen2.5-72b-chat"
)

# Configuration for evaluation model
EVAL_MODEL_PATH="/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
EVAL_MODEL_NAME="qwen2.5-72b-chat"

# Common configuration
API_BASE="http://localhost:8010/v1"
PORT=8010
GPU=4
THREADS=30

# Input/Output paths
INPUT_FILES="dataset/fantiasic_logic_puzzles.jsonl,dataset/the_canterbury_puzzles_and_other_curious_problems.jsonl"
ADVICE_FILES="advice_0520_v3/qwen3-32b-chat_logic_puzzle_advice.jsonl,advice_0520_v3/qwen3-32b-chat_canterbury_puzzle_advice.jsonl"
# Create output directories if they don't exist
mkdir -p result_with_advice

# Step 1: Generate answers with all models
echo "Starting answer generation phase..."
for i in "${!model_paths[@]}"; do
    MODEL_PATH="${model_paths[$i]}"
    MODEL_NAME="${model_names[$i]}"
    ADVICE_FILE="${ADVICE_FILES[$i]}"
    
    # Set output paths for this model
    OUTPUT_FILES="result_with_advice/${MODEL_NAME}_logic_answers.jsonl,result_with_advice/${MODEL_NAME}_math_answers.jsonl"
    
    echo "Generating answers with ${MODEL_NAME}..."
    python generate_puzzle_answers_with_advice.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --api_base "$API_BASE" \
        --port $PORT \
        --gpu $GPU \
        --threads $THREADS \
        --input_file "$INPUT_FILES" \
        --output_file "$OUTPUT_FILES" \
        --advice_file "$ADVICE_FILE"
done

# Step 2: Evaluate all generated answers with 72B model
echo "Starting evaluation phase with ${EVAL_MODEL_NAME}..."

declare -a model_paths=(
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
)

declare -a model_names=(
    "qwen2.5-7b"
    "qwen2.5-32b-chat"
    "qwen2.5-14b-chat"
    "qwen2.5-72b-chat"
)
# Collect all output files for evaluation
ALL_OUTPUT_FILES=""
ALL_EVAL_FILES=""
for model_name in "${model_names[@]}"; do
    ALL_OUTPUT_FILES+="result_with_advice/${model_name}_logic_answers.jsonl,result_with_advice/${model_name}_math_answers.jsonl,"
    ALL_EVAL_FILES+="result_with_advice_eval/${model_name}_logic_eval.jsonl,result_with_advice_eval/${model_name}_math_eval.jsonl,"
done
# Remove trailing comma
ALL_OUTPUT_FILES=${ALL_OUTPUT_FILES%,}
ALL_EVAL_FILES=${ALL_EVAL_FILES%,}

echo "Evaluating all generated answers..."
python eval_puzzle_answers.py \
    --model_path "$EVAL_MODEL_PATH" \
    --model_name "$EVAL_MODEL_NAME" \
    --api_base "$API_BASE" \
    --port $PORT \
    --gpu $GPU \
    --threads $THREADS \
    --path_to_jsonl_list "$ALL_OUTPUT_FILES" \
    --output_file_list "$ALL_EVAL_FILES"

echo "All pipelines completed!" 


# cd /home/aiscuser/zhengyu_blob_home/tony_folder/0104_three_type/lm-evaluation-harness

# bash hpc_leaderboard_0514.sh


bash /home/aiscuser/zhengyu_blob_home/kkk_vllm_first_4.sh