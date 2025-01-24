#!/bin/bash

GPU_ID=0
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu_id)
      GPU_ID="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    *)
      echo "Unknow $1"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model_path REQUIRED！"
  exit 1
fi

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" python chat.py --model_path "$MODEL_PATH"
