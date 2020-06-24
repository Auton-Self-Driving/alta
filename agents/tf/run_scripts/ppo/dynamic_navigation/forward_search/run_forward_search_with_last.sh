# ssh gpu21

python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 6000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 1 &

python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 6000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 2 &

python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 6000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 3 &