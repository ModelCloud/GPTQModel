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
    sha256=$(sha256sum dist/$whl)
    response=$(curl -s -F "runid=$RUN_ID" -F "repo=${{ env.repo }}" -F "ref=${{ env.ref }}" -F "sha256=$sha256" -F "file=@dist/${{ env.WHL_NAME }}" http://${{ needs.check-vm.outputs.ip }
    }}/gpu/whl/upload)
    if [ "$response" -eq 0 ]; then
      echo "UPLOADED=1" >> $GITHUB_ENV
    fi
}

cuda=$1
torch=$2
python=$3
run_id=$4
repo=$5
ref=$6

echo "CUDA Version: $cuda"
export CUDA=$cuda
echo "Torch Version: $torch"
export TORCH=$torch
echo "Python Version: $python"
export PYTHON=$python
echo "Run id: $run_id"
export RUN_ID=$run_id
echo "Repo: $repo"
export REPO=$repo
echo "Ref: $ref"
export REF=$ref

install_requirements
compile
find_gpu
test || true
release_gpu
clear_cache
