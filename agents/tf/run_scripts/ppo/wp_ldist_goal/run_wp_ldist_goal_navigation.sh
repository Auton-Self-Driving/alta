python run_code.py \
--algo PPO \
--input-type wp_ldist_goal \
--network 2_layer \
--scenarios navigation \
--timesteps 1000000 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light' \
--disable-traffic-light \
--disable-obstacle-info \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--run-id 1 &
sleep 60