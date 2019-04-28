#! /bin/bash

trap "kill 0" EXIT

env="Carla-9-4"    
CUDA_VISIBLE_DEVICES=0 python ../train/torch/train.py \
        --algo DDPG \
        --actor-lr 5e-6 \
        --critic-lr 5e-6 \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 4500 \
        --env-name $env \
        --start-steps 0 \
        --max-steps 300000 \
        --replay-size 50000 \
        --fixed-replay \
        --batch-size 256 \
        --optim "Adam" \
        --action_type "steer_only" \
        --log-interval 100 \
        --log-dir ../logs/$env/04_19 \
        --save-dir ../weights/$env/04_19 \
        --discount 0.99 \
        --file-name DDPG-ALR5e-6-CLR5e-6-steer-only-fixed-replay-50k

# wait
