sleep 360
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3/' \
--scenarios long_straight --val-trials 6 \
--expert-buffer-path '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1_runid_1/dqn_measurements_weights_buffer_latest.zip' \
--reduce-filename --expert-data-sample-percent 50 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 1 --videos >> expert_long_junc_fs3.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3/' \
--scenarios long_straight --val-trials 6 \
--expert-buffer-path '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1_runid_1/dqn_measurements_weights_buffer_latest.zip' \
--reduce-filename --expert-data-sample-percent 50 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 --videos >> expert_long_junc_fs3.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3/' \
--scenarios long_straight --val-trials 6 \
--expert-buffer-path '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_10_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_10_r_norm_24_suc_r_100_const_r_1_runid_1/dqn_measurements_weights_buffer_latest.zip' \
--reduce-filename --expert-data-sample-percent 50 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 10 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 1 --videos >> expert_long_junc_fs3.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3/' \
--scenarios long_straight --val-trials 6 \
--expert-buffer-path '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_10_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_vecs_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_10_r_norm_24_suc_r_100_const_r_1_runid_1/dqn_measurements_weights_buffer_latest.zip' \
--reduce-filename --expert-data-sample-percent 50 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 10 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 --videos >> expert_long_junc_fs3.txt 2>&1 &
