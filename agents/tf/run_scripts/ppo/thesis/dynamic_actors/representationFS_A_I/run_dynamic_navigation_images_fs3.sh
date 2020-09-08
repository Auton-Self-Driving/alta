# NOTE: Enable camera coordinates from environment file (env.py)
python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/forward_facing_ss/' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--train-vae --ae-lr 5e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/forward_facing_ss/' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--train-vae --ae-lr 5e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/forward_facing_ss/' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--train-vae --ae-lr 5e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/forward_facing_ss/' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--train-vae --ae-lr 5e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 4 &
sleep 60