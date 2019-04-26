#! /bin/bash

trap "kill 0" EXIT

env="Carla-9-4"    
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python ../train/torch/train.py \
        --algo DDPG \
        --pixel-model "vae" \
        --actor-lr 1e-5 \
        --critic-lr 1e-5 \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 3906 \
        --env-name $env \
        --start-steps 0 \
        --max-steps 300000 \
        --replay-size 50000 \
        --fixed-replay \
        --batch-size 256 \
        --log-interval 100 \
        --log-dir ../logs/$env/04_19 \
        --save-dir ../weights/$env/04_19 \
        --discount 0.99 \
        --file-name DDPG-ALR1e-5-CLR1e-5-vae-fixed-replay-50k

# wait
