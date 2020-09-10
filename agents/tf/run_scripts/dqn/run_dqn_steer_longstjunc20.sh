
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 10 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 10 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &


sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 10 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 10 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 15 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 15 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &


sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 15 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --special-sample --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 75 --constant-reward 1 --dqn-n-step 15 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs1_ss_nn128_trainfreq_4.txt 2>&1 &

