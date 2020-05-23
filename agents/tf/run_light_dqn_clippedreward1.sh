python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 1 --videos &

sleep 120

python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 2 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos &

sleep 120

python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 --videos &

sleep 120

python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 3e-6 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 --videos &


sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 --videos &

sleep 120

python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 --videos &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--base-log-dir '/home/scratch/hiteshar/research/alta-logs/dqn_new_light_clippedReward_runs_1/longst_steer_throttle_1/' \
--scenarios long_straight \
--timesteps 10000000 \
--num-npc 50 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs --clip-reward \
--const-collision-penalty 3  --steer-penalty-coeff  0 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 --videos &
