#!/bin/bash
export RUNNER=10.0.14.248

function install_requirements() {
  bash -c "$(curl -L http://$RUNNER/scripts/compiler/init_env.sh)" @ $CUDA $TORCH $PYTHON
  uv pip install auto_round optimum bitblas==0.0.1.dev13 parameterized uvicorn -i http://$RUNNER/simple/ --trusted-host $RUNNER
  uv pip install transformers -U -i http://$RUNNER/simple/ --trusted-host $RUNNER
}

function compile() {
    python setup.py bdist_wheel

    ls -ahl dist

    whl=$(ls -t dist/*.whl | head -n 1 | xargs basename)
    sha256=$(sha256sum dist/$whl)
    echo "hash=$sha256"

    twine check dist/$whl
    pip install dist/$whl

    # upload to artifact
    # xxx
}

function clear_cache() {
  if [ "$ERROR" == "1" ] && [ "$BITBLAS" == "1" ] && [ "$ERROR" == "1" ]; then
      rm -rf ~/.cache/bitblas/nvidia/geforce-rtx-4090
      echo "clear bitblas cache"
  fi
}

function find_gpu() {
  timestamp=$(date +%s%3N)
  gpu_id=-1

  while [ "$gpu_id" -lt 0 ]; do
    gpu_id=$(curl -s "http://$RUNNER/gpu/get?id=$RUN_ID&timestamp=$timestamp")

    if [ "$gpu_id" -lt 0 ]; then
      echo "http://$RUNNER/gpu/get?id=$RUN_ID&timestamp=$timestamp returned $gpu_id"
      echo "No available GPU, waiting 5 seconds..."
      sleep 5
    else
      echo "Allocated GPU ID: $gpu_id"
    fi
  done
  export CUDA_VISIBLE_DEVICES=$gpu_id
  export STEP_TIMESTAMP=$timestamp
  echo CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES, timestamp=$STEP_TIMESTAMP
}

function release_gpu() {
    curl -X GET "http://$RUNNER/gpu/release?id=$RUN_ID&gpu=$CUDA_VISIBLE_DEVICES&timestamp=$STEP_TIMESTAMP"
}

function test() {
  echo "current dir:"
  pwd
  echo "===="
  ls
  echo "===="
  ls ..
  echo "===="
  pytest --durations=0 tests/$TEST_NAME.py || { export ERROR=1; exit 1; }
}

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <test_name> <cuda> <torch> <python> <run_id> <docker>"
    exit 1
fi

test_name=$1
cuda=$2
torch=$3
python=$4
run_id=$5
docker=$6

echo "Test Name: $test_name"
export TEST_NAME="${test_name%.py}"
echo "CUDA Version: $cuda"
export CUDA=$cuda
echo "Torch Version: $torch"
export TORCH=$torch
echo "Python Version: $python"
export PYTHON=$python
echo "Run id: $run_id"
export RUN_ID=$run_id
echo "Docker image: $docker"
export DOCKER_IMAGE=$docker

install_requirements
compile
find_gpu
test || true
release_gpu
clear_cache
