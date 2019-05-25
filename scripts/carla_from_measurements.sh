#! /bin/bash

trap "kill 0" EXIT

env="Carla-9-4"
alr=1e-6 #1e-5
clr=1e-5
lr=1e-5
action="steer_only"
reward="corlA"
optim="Adam"
cnn="Smallest"
scenarios="straight"
obs="measurement"
CUDA_VISIBLE_DEVICES=1 python ../train/torch/train.py \
        --algo DDPG \
        --actor-lr $alr \
        --critic-lr $clr \
        --target-lr $lr \
        --dropout-rate 0.0 \
        --pretrained none \
        --carla-port 9500 \
        --env-name $env \
        --start-steps 5000 \
        --max-steps 300000 \
        --replay-size 100000 \
        --fixed-replay \
        --cnn-size $cnn \
        --batch-size 256 \
        --optim $optim \
        --action-type $action \
        --reward-function $reward \
        --log-interval 100 \
        --log-dir ../logs/$env/05_25 \
        --save-dir ../weights/$env/05_25 \
        --discount 0.99 \
        --scenarios $scenarios \
        --obs-space $obs \
        --file-name DDPG-ALR$alr-CLR$clr-$action-$reward-measurements-only-100k-run_right_steeronly_1

# wait
