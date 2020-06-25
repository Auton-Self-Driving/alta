python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128.txt 2>&1 &


sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 3 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 1 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128.txt 2>&1 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 --videos >> dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128.txt 2>&1 &


sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128/lstjunc_steer7_throttle_0_20_ac12/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs --target-freq 2000 \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 --reward-norm 8 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 --videos &

