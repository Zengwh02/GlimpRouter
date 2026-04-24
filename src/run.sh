#!/bin/bash
# Example runner; edit the parameters below for your experiment.
#################################################

CONFIG_PATH="config.json"
DATASET_NAME="aime25"
REPEAT_NUM=1
SCORE_METHOD="first_token_entropy"
TOKEN_BUDGET=8192
MODEL_SIZE="32b"
SMALL_MODEL_SIZE="4b"
SCORE_TRESHOLD=0.9
OUTPUT_DIR="./large${MODEL_SIZE}_small${SMALL_MODEL_SIZE}_${SCORE_METHOD}_${SCORE_TRESHOLD}_results"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/large${MODEL_SIZE}_small${SMALL_MODEL_SIZE}_${SCORE_METHOD}_${SCORE_TRESHOLD}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).out"

echo "Logging ${LOG_FILE}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Write the runtime config consumed by main.py.
cat <<EOF > "${CONFIG_PATH}"
{
  "dataset_name": "${DATASET_NAME}",
  "repeat_num": ${REPEAT_NUM},
  "score_method": "${SCORE_METHOD}",
  "token_budget": ${TOKEN_BUDGET},
  "output_dir": "${OUTPUT_DIR}",
  "model_size": "${MODEL_SIZE}",
  "small_model_size": "${SMALL_MODEL_SIZE}",
  "score_threshold": ${SCORE_TRESHOLD}
}
EOF

nohup python main.py > ${LOG_FILE} 2>&1 
