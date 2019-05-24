#! /bin/bash

trap "kill 0" EXIT

env="Carla-9-4"
alr=1e-5
clr=1e-4
lr=1e-4
action="merged_gas"
reward="new2"
optim="Adam"
cnn="Smallest"
scenarios="straight"
CUDA_VISIBLE_DEVICES=1 python ../train/torch/train.py \
        --algo DDPG \
        --actor-lr $alr \
        --critic-lr $clr \
        --target-lr $lr \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 9500 \
        --env-name $env \
        --start-steps 10000 \
        --max-steps 300000 \
        --replay-size 100000 \
        --fixed-replay \
        --cnn-size $cnn \
        --batch-size 256 \
        --optim $optim \
        --action-type $action \
        --reward-function $reward \
        --log-interval 100 \
        --log-dir ../logs/$env/05_20 \
        --save-dir ../weights/$env/05_20 \
        --discount 0.99 \
        --scenarios $scenarios \
        --file-name DDPG-ALR$alr-CLR$clr-$scenarios-$action-$reward-orig-rgb-image-fixed-replay-100k-w-reward-scaling-notraining-256

# wait
