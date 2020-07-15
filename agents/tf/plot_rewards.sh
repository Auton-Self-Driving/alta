
# gpu5
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_fixed/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained, fs3, n=3,uniform)' \
--inds 1 2 3


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs3, n=3,uniform)' \
--inds 1 2 

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_fs_3_dqn_n_10_r_norm_24_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs3, n=10,uniform)' \
--inds 1 2 

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3,uniform)' \
--inds 1 2 3

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_dqn_n_3_rew_norm_8_successr_100_constantr_1' \
--title 'Long Junc (fs1, n=3,uniform)' \
--inds 1 2 3

# gpu6
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_ss_fixed/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained, fs3, n=3, Backward Sample)' \
--inds 1 2 3


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs1' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_dqn_n_10_r_norm_8_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs1, n=10, lr=1e-5,uniform)' \
--inds 1 2 

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs1' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_dqn_n_10_r_norm_8_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs1, n=10, lr=3e-5,uniform)' \
--inds 1 2 


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5,ss)' \
--inds 1 2 3

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_ss_dqn_n_3_rew_norm_8_successr_100_constantr_1' \
--title 'Long Junc (fs1, n=3, lr=3e-5,ss)' \
--inds 1 2 3

# gpu4
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_ss_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, fs3, n=3, Backward Sample)' \
--inds 1 2 3

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, fs3, n=3, uniform)' \
--inds 1 2 3


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_500_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5, target=500, ss)' \
--inds 1 2 3

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_500_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5, target=500, uniform)' \
--inds 1 2 3


#gpu16

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_1' \
--run-path 'algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp angles, expert 25%, fs3, n=3, lr=3e-5,ss)' \
--inds 1 2 3


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_uniform_1' \
--run-path 'algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp angles, expert 25%, fs3, n=3, lr=3e-5,uniform)' \
--inds 1 2 3

# gpu9
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 25%, fs3, n=3, lr=3e-5,ss)' \
--inds 1 2 3


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_uniform_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp angles, expert 25%, fs3, n=3, lr=3e-5,uniform)' \
--inds 1 2 3


algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1_runid_1/test_results.csv