#! /bin/bash

trap "kill 0" EXIT

env="Carla-9-4"
lr=1e-5
action="steer_only"
reward="simplest"
optim="Adam"
CUDA_VISIBLE_DEVICES=0 python ../train/torch/train.py \
        --algo DDPG \
        --actor-lr $lr \
        --critic-lr $lr \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 4500 \
        --env-name $env \
        --start-steps 0 \
        --max-steps 300000 \
        --replay-size 50000 \
        --fixed-replay \
        --batch-size 256 \
        --optim $optim \
        --action-type $action \
        --reward-function $reward \
        --log-interval 100 \
        --log-dir ../logs/$env/04_19 \
        --save-dir ../weights/$env/04_19 \
        --discount 0.99 \
        --file-name DDPG-ALR$lr-CLR$lr-$action-$reward-fixed-replay-50k

# wait
