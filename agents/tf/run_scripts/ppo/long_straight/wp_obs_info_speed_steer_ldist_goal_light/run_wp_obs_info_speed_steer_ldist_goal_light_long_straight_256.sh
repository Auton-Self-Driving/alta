python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--run-id 3 &
sleep 60









python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--videos \
--run-id 3 &
sleep 60










python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 500 --collision-penalty-speed-coeff 500 \
--const-light-penalty 500 --light-penalty-speed-coeff 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 500 --collision-penalty-speed-coeff 500 \
--const-light-penalty 500 --light-penalty-speed-coeff 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--const-collision-penalty 500 --collision-penalty-speed-coeff 500 \
--const-light-penalty 500 --light-penalty-speed-coeff 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_256' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--run-id 3 &
sleep 60