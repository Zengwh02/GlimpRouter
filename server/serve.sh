#!/bin/bash
# Starts a vLLM server and captures logs; fill placeholders before running.
LOG_DIR="vllm_logs"  # or any directory you prefer

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

#################################################

# Starting the LARGE model server

CUDA_DEVICE=YOUR_CUDA_DEVICE_ID  # NOTE: change to your CUDA device id, e.g. 0/1/2/3
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # OR change to your model path/name, e.g. org/model-name
API_KEY="glimp_router"  # OR change to the API key of your model (use a placeholder if auth is disabled)

LOG_FILE="${LOG_DIR}/CUDA${CUDA_DEVICE}_server_$(date +%Y%m%d_%H%M%S).out"
TEMPLATE_FILE="./template/deepseekr1.jinja"  # OR change to the template file of your model, e.g. "./template/your_model.jinja"
GPU_MEMORY_UTILIZATION=0.60  # OR change to the gpu memory utilization of your model, e.g. 0.60

PORT=YOUR_PORT  # NOTE: change to the port of your model, e.g. 11125

pid=$(lsof -ti tcp:$PORT)

if [ -n "$pid" ]; then
  # Free the port if another process is already listening.
  echo "Port $PORT is occupied, process PID: $pid"
  echo "Kill the process..."
  kill -9 $pid
fi

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup vllm serve $MODEL \
  --dtype auto \
  --max-model-len 16384 \
  --chat-template $TEMPLATE_FILE \
  --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
  --api-key $API_KEY \
  --port $PORT \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  > $LOG_FILE 2>&1 &


#################################################

# Starting the SMALL model server

CUDA_DEVICE=YOUR_CUDA_DEVICE_ID  # NOTE: change to your CUDA device id, e.g. 0/1/2/3
MODEL="Qwen/Qwen3-4B-Thinking-2507"  # OR change to your model path/name, e.g. org/model-name
API_KEY="glimp_router"  # OR change to the API key of your model (use a placeholder if auth is disabled)

LOG_FILE="${LOG_DIR}/CUDA${CUDA_DEVICE}_server_$(date +%Y%m%d_%H%M%S).out"
TEMPLATE_FILE="./template/qwen3.jinja"  # OR change to the template file of your model, e.g. "./template/your_model.jinja"
GPU_MEMORY_UTILIZATION=0.30  # OR change to the gpu memory utilization of your model, e.g. 0.60

PORT=YOUR_PORT  # NOTE: change to the port of your model, e.g. 11125

pid=$(lsof -ti tcp:$PORT)

if [ -n "$pid" ]; then
  # Free the port if another process is already listening.
  echo "Port $PORT is occupied, process PID: $pid"
  echo "Kill the process..."
  kill -9 $pid
fi

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup vllm serve $MODEL \
  --dtype auto \
  --max-model-len 16384 \
  --chat-template $TEMPLATE_FILE \
  --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
  --api-key $API_KEY \
  --port $PORT \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  > $LOG_FILE 2>&1 &
