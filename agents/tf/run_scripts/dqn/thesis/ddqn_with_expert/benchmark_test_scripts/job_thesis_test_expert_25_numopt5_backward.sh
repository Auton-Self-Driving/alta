
# No Crash Dense

python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 100 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 100 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 100 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 70 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 70 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_dense --val-trials 25 \
--num-npc 70 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &



# no_crash_regular
sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 

sleep 2000
sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &


python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_regular --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 200
# no_crash_empty
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 

sleep 1800
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios no_crash_empty --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1

sleep 2400
# dynamic_navigation
sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 20 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &


python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios dynamic_navigation --val-trials 25 \
--num-npc 15 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 120
# navigation
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &


sleep 100
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 

sleep 1800
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios navigation --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 200
# curved
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 100
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 

sleep 1800
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios curved --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &


# straight
sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 100
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town01' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_1/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_2/dqn_measurements_weights1480000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &

sleep 300
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/' \
--agent_model_path '/zfsauton2/home/hiteshar/research/alta-logs/dqn_expert_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_target2k/steer7_throttle_0_20_ac12/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1/algo_DQN_input_wp_obs_info_speed_steer_ldist_light_lr_1e-05_expert_25.0_exp_0.1_dynamic_navigation_npc_50_steer_pen_2.0_fs_2_ss_dqn_n_3_r_norm_16_suc_r_75_const_r_1_runid_3/dqn_measurements_weights1560000.zip' \
--timesteps 10000000 \
--scenarios straight --val-trials 25 \
--num-npc 0 \
--disable-sample-npc \
--test \
--test-trails 3 \
--disable-lane-invasion --light-thresold 15  --min-light-thresold 6 --disable-collision \
--disable-traffic-light-termination \
--city_name 'Town02' \
--enable-brake \
--disable-semantic  \
--special-sample --use-pid-fs --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_dynamic_nav_steer7_speed_0_20_ac12_fs3_ss_nn256_128_64_test3.txt 2>&1 &


