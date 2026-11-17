#!/bin/bash
# Wait until CUDA works again on this machine, then run the given sweep.
# usage: bash new_wamal/wait_and_run.sh E0 0,1,3
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/rl-aux/bin/python
until CUDA_VISIBLE_DEVICES=${2%%,*} $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; do
  echo "$(date) CUDA unavailable; checking again in 10 min"; sleep 600
done
echo "$(date) CUDA is back; starting sweep $1 on GPUs $2"
exec nice -n 10 $PY -m new_wamal.sweep --exp "$1" --gpus "$2"
