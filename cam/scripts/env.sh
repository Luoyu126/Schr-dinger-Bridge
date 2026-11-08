#!/bin/bash
# Environment for running the vendored CAM code (local addition, see cam/UPSTREAM.md).
# Source it:  source cam/scripts/env.sh
#
# - Uses the existing conda env ~/anaconda3/envs/CAM READ-ONLY (never pip/conda install into it).
# - Points every cache/config/temp dir under /usr0/home/shengyuf/yunyi/ and disables wandb,
#   so that nothing is written to ~/.cache, ~/.config, ~/.triton, ~/.nv, /tmp or the conda env.

_CAM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAM_ROOT="$(cd "$_CAM_SCRIPTS_DIR/.." && pwd)"     # .../Schr-dinger-Bridge/cam
export CAM_REPO="$(cd "$CAM_ROOT/.." && pwd)"             # .../Schr-dinger-Bridge
export CAM_CONDA_BIN="${CAM_CONDA_BIN:-/usr0/home/shengyuf/anaconda3/envs/CAM/bin}"
export CAM_PYTHON="$CAM_CONDA_BIN/python"
export CAM_ACCELERATE="$CAM_CONDA_BIN/accelerate"
export CAM_DATADIR="${CAM_DATADIR:-$CAM_REPO/data/cam}"   # gitignored; made by scripts/export_cam_dataset.py
export CAM_RUNDIR="${CAM_RUNDIR:-$CAM_REPO/runs/cam}"     # gitignored (runs/)

_CAM_CACHE=/usr0/home/shengyuf/yunyi/.cache/cam
export XDG_CACHE_HOME="$_CAM_CACHE"
export XDG_CONFIG_HOME="$_CAM_CACHE/config"
export HF_HOME="$_CAM_CACHE/hf"
export TORCH_HOME="$_CAM_CACHE/torch"
export TRITON_CACHE_DIR="$_CAM_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$_CAM_CACHE/inductor"
export PYG_HOME="$_CAM_CACHE/pyg"
export CUDA_CACHE_PATH="$_CAM_CACHE/nv"
export MPLCONFIGDIR="$_CAM_CACHE/mpl"
export TMPDIR="$_CAM_CACHE/tmp"
export WANDB_MODE=disabled
export WANDB_DIR="$_CAM_CACHE/wandb"
export WANDB_CONFIG_DIR="$_CAM_CACHE/wandb"
export WANDB_CACHE_DIR="$_CAM_CACHE/wandb"
export WANDB_DATA_DIR="$_CAM_CACHE/wandb"
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in cam/ or in the conda env
export PYTHONNOUSERSITE=1          # ignore ~/.local site-packages
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$XDG_CONFIG_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$PYG_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR" "$WANDB_DIR"

# Refuse to run unless CUDA_VISIBLE_DEVICES names the GPU(s) explicitly and none of them runs
# another user's process.
cam_require_gpu() {
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "error: set CUDA_VISIBLE_DEVICES to the GPU assigned to this run" >&2
        return 1
    fi
    # Refuse any GPU that currently runs another user's process (GPU 1 used to be another
    # user's; it may be used only while it is free).
    local me g uuid pid owner
    me=$(id -un)
    for g in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
        uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g {print $2}')
        for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
            owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
                echo "error: GPU $g runs a process of user $owner; do not use it" >&2; return 1
            fi
        done
    done
}
