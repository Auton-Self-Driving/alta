# python run_code.py \
# --algo DQN \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/' \
# --agent_model_path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/dqn_measurements_weights1000000.zip' \
# --scenarios long_straight \
# --timesteps 10000000 \
# --num-npc 50 \
# --enable-brake \
# --disable-semantic  \
# --use-pid-fs  \
# --const-collision-penalty 3  --steer-penalty-coeff  0 \
# --const-light-penalty 3 --light-penalty-speed-coeff 3 \
# --collision-penalty-speed-coeff 3 \
# --fs 1 \
# --lr 3e-5 --buffer-size 1000000 \
# --carla-gpu 3 --code-gpu 3 \
# --run-id 1 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/' \
--agent_model_path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-06_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-06_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/dqn_measurements_weights1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs  \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 1 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/' \
--agent_model_path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-06_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-06_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/dqn_measurements_weights1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs  \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 1 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/' \
--agent_model_path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/dqn_measurements_weights1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs  \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos &
