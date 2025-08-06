#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="/data/minimax-dialogue/feishan/models/gpt-oss-120b"  # 替换为你的模型路径
OUTPUT_FILE="outputs_gpt-oss-120b_high.json"        # 输出文件名
REASONING_EFFORT="high"

python eval_mmlupro.py \
  --model_path "$MODEL_PATH" \
  --output_file "$OUTPUT_FILE" \
  --reasoning_effort "$REASONING_EFFORT"
