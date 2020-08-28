python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/dqn_measurements_weights_buffer_1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 1000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id test1 &


python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/dqn_measurements_weights_buffer_1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 1000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id test2 --train-from-scratch


python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--agent_model_path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_pre_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_pre_npc_50_buffer_1000000_brake_target_freq_1000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/dqn_measurements_weights_buffer_1000000.zip' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 1000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id test2 --val-trials 6