
# gpu5
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_fixed/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained, fs3, n=3,uniform)' \
--save-auton --inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs3, n=3,uniform)' \
--save-auton --inds 1 2 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs3' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_fs_3_dqn_n_10_r_norm_24_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs3, n=10,uniform)' \
--save-auton --inds 1 2 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3,uniform)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_dqn_n_3_rew_norm_8_successr_100_constantr_1' \
--title 'Long Junc (fs1, n=3,uniform)' \
--save-auton --inds 1 2 3 &

# gpu6
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_ss_fixed/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained, fs3, n=3, Backward Sample)' \
--save-auton --inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs1' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_dqn_n_10_r_norm_8_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs1, n=10, lr=1e-5,uniform)' \
--save-auton --inds 1 2 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_long_junc/expert_long_junc_fs1' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_3e-05_expert_50.0_exp_0.1_long_straight_npc_50_target_freq_10000_steer_pen_2.0_dqn_n_10_r_norm_8_suc_r_100_const_r_1_optep_1' \
--title 'Long Junc (expert, fs1, n=10, lr=3e-5,uniform)' \
--save-auton --inds 1 2 &


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_ss_dqn_n_3_rew_norm_8_successr_100_constantr_1' \
--title 'Long Junc (fs1, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/cddqn_lstjunc_nav_steer7_speed_0_20_ac12_fs2_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'Long Junc (CDDQN, fs=2 n=3, lr=1e-5,ss)' \
--save-auton --inds 1 2 3 &


# gpu4
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_ss_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, fs3, n=3, Backward Sample)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, fs3, n=3, uniform)' \
--save-auton --inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_ss_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_500_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5, target=500, ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_target_freq_500_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1' \
--title 'Long Junc (fs3, n=3, lr=3e-5, target=500, uniform)' \
--save-auton --inds 1 2 3 &


#gpu16
/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu16/research/alta-logs/autobot-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/test_results.csv

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/autobot-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1' \
--title 'DN (wp angles, expert 25%, fs2, n=3, lr=1e-5,ss, target=2k)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/autobot-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1' \
--title 'DN (wp angles, expert 50%, fs2, n=3, lr=1e-5,ss, target=2k)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_1' \
--run-path 'algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp angles, expert 25%, fs3, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_uniform_1' \
--run-path 'algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp angles, expert 25%, fs3, n=3, lr=3e-5,uniform)' \
--save-auton --inds 1 2 3 &

# gpu9
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 25%, fs3, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_uniform_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 25%, fs3, n=3, lr=3e-5,uniform)' \
--save-auton --inds 1 2 3 &

# gpu8
python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 50%, fs3, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_uniform_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 50%, fs3, n=3, lr=3e-5,uniform)' \
--save-auton --inds 1 2 3 &



algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_angles_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1_runid_1/test_results.csv
# carla
//algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_long_straight_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1_runid_1/test_results.csv

python plot_rewards.py \
--log-path '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_long_straight_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'Long Junc (fs2, n=3, nn:256, 128, 64, lr=1e-5, target=10k, ss)' \
--save-auton --inds 1 2 


#autobot

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_nav_ac12_dis_light_dis_lane_termination_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_navigation_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dis_light_dis_lane_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'Nav w/o actors  (scratch, fs3, n=3, ss)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_nav_ac12_dis_light_dis_lane_termination_fs1/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_navigation_pre_tfs_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_dis_light_dis_lane_dqn_n_3_r_norm_8_suc_r_100_const_r_1' \
--title 'Nav w/o actors (pretrained on tjunc-light, fs1, n=3, ss)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_nav_ac12_dis_light_dis_lane_termination_scratch_fs1/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_navigation_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_pid_fs_dis_light_dis_lane_dqn_n_3_r_norm_8_suc_r_100_const_r_1' \
--title 'Nav w/o actors  (scratch, fs1, n=3, ss)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_re/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained on tjunc-light, fs3, n=3, ss)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_re/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_dynamic_navigation_pre_tfs_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (pretrained on tjunc-light, fs3, n=3, uniform)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_nav_ac12_dis_light_dis_lane_termination/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_navigation_pre_tfs_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dis_light_dis_lane_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'Nav w/o actors (pretrained on tjunc-light, fs3, n=3, ss)' \
--inds 1 2 3 &


# latest runs
python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_per_nn256_128_64_v2/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fsper_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, per)' \
--inds 1 2 3 &

# expert experiments

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 25, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 50, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_100.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 100, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 25, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 50, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_100.0_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (expert 100, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_target_freq_5000_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_2' \
--title 'DN (expert 25, nn256-128-64, fs2, n=3, ss, target5k)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_target_freq_5000_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_2' \
--title 'DN (expert 50, nn256-128-64, fs2, n=3, ss, target5k)' \
--inds 1 2 3 &


ccddqn

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/cddqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_cDDQN_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN (scratch, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/cddqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN (scratch, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/ccdqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_per_nn256_128_64_v2/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_cDDQNper_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN(scratch, nn256-128-64, fs2, n=3, per)' \
--inds 1 2 3 &


# gpu4 ones running from scratch

dqn_dynamic_nav_improv_large_steer_ac12_uniform_scratch

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_uniform_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, nn126-128, fs3, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_improv_large_steer_ac12_ss_scratch/junc_steer7_throttle_0_20_ac_12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_dynamic_navigation_npc_70_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (scratch, nn128-128, fs3, n=3, ss)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/home/hiteshar/alta-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_per_nn256_128_64_v2/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fsper_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn128-128, fs3, n=3, per)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/media/hdd/hiteshar/autobot-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/media/hdd/hiteshar/autobot-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/media/hdd/hiteshar/autobot-logs/dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_per_nn256_128_64_v2/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fsper_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN (scratch, nn256-128-64, fs2, n=3, per)' \
--inds 1 2 3 &


python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/autobot-logs/cddqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_uniform_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_cDDQN_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN (scratch, nn256-128-64, fs2, n=3, uniform)' \
--inds 1 2 3 &

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/autobot-logs/cddqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN (scratch, nn256-128-64, fs2, n=3, ss)' \
--inds 1 2 3 &



python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/autobot-logs/ccdqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_per_nn256_128_64_v2/steer7_throttle_0_20_ac12' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_target_freq_10000_steer_pen_2.0_fs_2_cDDQNper_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--title 'DN with CDDQN(scratch, nn256-128-64, fs2, n=3, per)' \
--inds 1 2 3 &


/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu8/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_wp_orienation/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_50.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1_runid_3/test_results.csv

python plot_rewards.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/expert_dynamic_nav_nn128_128/expert_dynamic_nav_nn128_128_fs3_ss_wp_orienation' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_3_ss_dqn_n_3_r_norm_24_suc_r_100_const_r_1' \
--title 'DN (wp orientation, expert 25%, fs3, n=3, lr=3e-5,ss)' \
--save-auton --inds 1 2 3 &


python plot_combined.py \
--log-path '/home/hiteshar/alta-logs/thesis_dqn_results' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate (DDQN Agent)' \
--reward-title 'Cumulative Reward (DDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


python plot_combined.py \
--log-path '/home/hiteshar/alta-logs/thesis_cddqn_results' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


python plot_combined_expert.py \
--log-path '/home/hiteshar/alta-logs/thesis_expert_uniform' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'DDQN Agent with Expert Demonstrations - Uniform Sampling' \
--reward-title 'DDQN Agent with Expert Demonstrations - Uniform Sampling' \
--fs 2 \
--inds 1 2 3 &

python plot_combined_expert.py \
--log-path '/home/hiteshar/alta-logs/thesis_expert_backward' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'DDQN Agent with Expert Demonstrations - Backward Sampling' \
--reward-title 'DDQN Agent with Expert Demonstrations - Backward Sampling' \
--fs 2 \
--inds 1 2 3 &

python plot_combined_expert_target.py \
--log-path '/home/hiteshar/alta-logs/thesis_expert50_target' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'DDQN-Expert-50 Agent: Affect of Optimization Epochs' \
--reward-title 'DDQN-Expert-50 Agent: Affect of Optimization Epochs' \
--fs 2 \
--inds 1 2 3 &

python plot_combined_expert_target.py \
--log-path '/home/hiteshar/alta-logs/thesis_expert25_target' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'DDQN-Expert-25 Agent: Affect of Optimization Epochs' \
--reward-title 'DDQN-Expert-25 Agent: Affect of Optimization Epochs' \
--fs 2 \
--inds 1 2 3 &


# gpu3 longst
python plot_combined_longst.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/thesis_longst' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate of DDQN Agent in Straight-Dynamic Task' \
--reward-title 'Cumulative Reward of DDQN Agent in Straight-Dynamic Task' \
--fs 5 \
--inds 1 2 3 &


python plot_combined.py \
--log-path '/home/hiteshar/alta-logs/thesis_cddqn_results' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


gpu6/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/Validation_qvalue_plots_buffer_old/E_1080_t_390000_i_0_v_40_qvalues_array.npz


python plot_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/Validation_qvalue_plots_buffer_old/' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &

/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu19/research/alta-logs/E_0_t_0_i_0_v_1_qvalues_array.npz
/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu19/research/alta-logs/dqn_replayRL1_wo_buffer1/longst_steer_throttle_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_3/buffer_training/Validation_qvalue_plots_buffer/E_0_t_0_i_0_v_1_qvalues_array.npz

/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu19/research/alta-logs/dqn_replayRL1_wo_buffer1/longst_steer_throttle_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_3/buffer_training/Validation_qvalue_plots_buffer/E_0_t_0_i_0_v_1_qvalues_array.npz
dqn_replayRL1_wo_buffer1/longst_steer_throttle_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_3/buffer_training/Validation_qvalue_plots_buffer
python plot_average_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_replayRL1_wo_buffer1/longst_steer_throttle_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_3/buffer_training/Validation_qvalue_plots_buffer/' \
--run-path 'dqn_replayRL1_wo_buffer1/longst_steer_throttle_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_3/buffer_training/Validation_qvalue_plots_buffer/' \
--success-title 'Q-Value and Discounted Returns Across Training' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &

python plot_average_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_long_straight_replay3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_use_pid_fs__runid_1/Validation_qvalue_plots_buffer/' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


/run/user/1000/gvfs/sftp:host=lop2.autonlab.org,user=hiteshar/zfsauton2/home/hiteshar/local_scratch5/gpu5/
python plot_average_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/Validation_qvalue_plots_buffer/' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Average Episodic Q-values over Training' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


python plot_average_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/cddqn_lstjunc_nav_steer7_speed_0_20_ac12_fs2_ss_nn256_128_64/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1_runid_1/Validation_qvalue_plots_buffer/' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Average Episodic Q-values over Training' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


python plot_average_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/cddqn_lstjunc_nav_steer7_speed_0_20_ac12_fs2_ss_nn256_128_64/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1_runid_3/Validation_qvalue_plots_buffer/' \
--run-path 'algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_1e-05_exp_0.1_dynamic_navigation_npc_50_brake_target_freq_10000_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_2_pid_fs_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1' \
--success-title 'Average Episodic Q-values over Training' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &


Average Episodic Q-values over Training

Avg Q-Values
Avg Discounted Return

E_38911_t_4523927_i_0_v_229_qvalues_array.npz

E_20372_t_1060953_i_0_v_55_qvalues_array.npz

python plot_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/Validation_qvalue_plots_buffer/' \
--run-path 'E_38911_t_4523927_i_0_v_229_qvalues_array.npz' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &

research/alta-logs/cddqn_lstjunc_nav_steer7_speed_0_20_ac12_fs2_ss_nn256_128_64/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_exp_0.1_long_straight_npc_10_target_freq_10000_steer_pen_2.0_fs_2_cDDQN_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_optep_1_runid_1/Validation_qvalue_plots_buffer/E_42542_t_8229315_i_0_v_301_qvalues_array.npz

python plot_qvalues.py \
--log-path '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_steer_pen_2.0_fs_3_pid_fs_dqn_n_3_rew_norm_24_successr_100_constantr_1_runid_1/Validation_qvalue_plots_buffer/' \
--run-path 'E_34762_t_3087915_i_0_v_157_qvalues_array.npz' \
--success-title 'Success Rate (cDDQN Agent)' \
--reward-title 'Cumulative Reward (cDDQN Agent)' \
--fs 2 \
--inds 1 2 3 &
