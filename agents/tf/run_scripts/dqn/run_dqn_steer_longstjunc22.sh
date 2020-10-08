python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --dqn-prioritized-replay --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --dqn-prioritized-replay --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/media/hdd/hiteshar/new-alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight --val-trials 6 \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --dqn-prioritized-replay --target-freq 10000 --opt-epochs 1 --exp-final-eps 0.1 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 2 --reward-norm 16 --success-reward 75 --constant-reward 1 --dqn-n-step 3 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 3 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_fs3_per_nn256_128_64.txt 2>&1 &
