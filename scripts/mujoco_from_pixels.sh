#! /bin/bash

trap "kill 0" EXIT

gpu=0

# for env in InvertedPendulum-v2 InvertedDoublePendulum-v2 HalfCheetah-v2 Pong-v0 CarRacing-v0
for env in Pong-v0
do
    
    # gpu=$(((gpu+1) % 3))

    CUDA_VISIBLE_DEVICES=0 python ../train/torch/train.py \
        --env-name $env \
        --algo DDPG \
        --start-steps 5000 \
        --max-steps 300000 \
        --actor-lr 1e-4 \
        --critic-lr 1e-4 \
        --file-name $env,DDPG,1e-4,from-pixels \
        --log-dir ../logs/$env/from-pixel1 \
        --pixel-obs
done