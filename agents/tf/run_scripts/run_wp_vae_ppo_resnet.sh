python ../run_code.py \
--algo PPO \
--input-type wp_resnet \
--network CustomPolicy2 \
--base-log-dir '~/alta-logs/new_env/ppo_runs/' \
--feat_extractor_model_path '~/alta/agents/tf/image_models/model-10.pth' \
--timesteps 1000 \
--carla-gpu 0 --code-gpu 1 \
--lr 2e-4 --run-id 1 &
sleep 60
