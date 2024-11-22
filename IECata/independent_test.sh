#!/bin/bash

module add CUDA/11.3

yhrun -N 1 -p v100 --gpus-per-node=1 --cpus-per-gpu=4 
  python independent_test_attention.py --cfg configs/kcatBAN.yaml --data kcat