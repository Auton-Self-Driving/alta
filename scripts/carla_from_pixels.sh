#! /bin/bash

trap "kill 0" EXIT

env="Carla-8-2"    
CUDA_VISIBLE_DEVICES=1 python ../train/torch/train.py \
        --algo DDPG \
        --actor-lr 3e-5 \
        --critic-lr 3e-5 \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 4500 \
        --env-name $env \
        --start-steps 0 \
        --max-steps 300000 \
        --replay-size 100000 \
        --fixed-replay \
        --batch-size 256 \
        --log-interval 100 \
        --log-dir ../logs/$env/04_01 \
        --save-dir ../weights/$env/04_01 \
        --discount 0.99 \
        --file-name DDPG-ALR3e-5-CLR3e-5-wo-dropout-fixed-replay-100k

# wait
