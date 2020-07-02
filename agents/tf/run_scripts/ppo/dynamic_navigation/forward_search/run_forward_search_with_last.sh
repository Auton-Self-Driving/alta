# ssh gpu21

python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last_updated' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 1 &

sleep 60
python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last_updated' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 2 &

sleep 120
python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/forward_search/low_dim_with_last_updated' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 3 &


sleep 300
python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight_junction \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_junction/forward_search/low_dim_with_last_updated' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 1 &

sleep 600
python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight_junction \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_junction/forward_search/low_dim_with_last_updated' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 2 &

sleep 900
python run_code.py \
--algo PPO \
--enable-search \
--pop-size 1 \
--pop-train-interval 120000 \
--disable-greedy-best \
--val-interval 120000 \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight_junction \
--timesteps 60000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_junction/forward_search/low_dim_with_last_updated' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 3 &