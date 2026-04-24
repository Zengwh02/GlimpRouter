#! /bin/bash

DATASET_NAME="aime25"
REPEAT_NUM=1
SCORE_METHOD="first_token_entropy"
SCORE_THRESHOLD=0.9

MODEL_PREFIX="large32b_small4b"  # NOTE: change to your customized text
ANSWER_PATH_PREFIX="../src/${MODEL_PREFIX}_${SCORE_METHOD}_${SCORE_THRESHOLD}_results/${DATASET_NAME}"  # NOTE: change to the path prefix of your answer file
OUTPUT_PATH_PREFIX="./xxx_results/${MODEL_PREFIX}_${SCORE_METHOD}_${SCORE_THRESHOLD}_${DATASET_NAME}"  # NOTE: change to the path prefix of your output file


for i in $(seq 1 $REPEAT_NUM); do
  python math_eval.py \
    --answer_path "${ANSWER_PATH_PREFIX}/result_${i}.json" \
    --dataset_name "${DATASET_NAME}" \
    --output_file "${OUTPUT_PATH_PREFIX}/result_${i}.json"
done