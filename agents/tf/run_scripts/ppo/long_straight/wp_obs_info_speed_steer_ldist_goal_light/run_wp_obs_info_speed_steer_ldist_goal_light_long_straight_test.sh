python run_code.py \
--algo PPO \
--test \
--test-trails 5 \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_runid_2/models/ppo2_weights6330000.pkl' \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--run-id 2 &
sleep 60