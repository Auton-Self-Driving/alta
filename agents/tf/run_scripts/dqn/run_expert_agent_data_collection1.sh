# python run_code.py \
# --algo DQN \
# --input-type wp_angles_vecs_obs_info_speed_steer_ldist_light \
# --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/' \
# --scenarios long_straight \
# --timesteps 50000 \
# --gen-expert-data \
# --num-npc 50 \
# --enable-brake \
# --disable-semantic  \
# --use-pid-fs \
# --const-collision-penalty 3  --steer-penalty-coeff  2 \
# --const-light-penalty 3 --light-penalty-speed-coeff 3 \
# --collision-penalty-speed-coeff 3 \
# --fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 3 \
# --lr 3e-5 --buffer-size 1000000 \
# --carla-gpu 3 --code-gpu 3 \
# --run-id 1 & 

# sleep 120
python run_code.py \
--algo DQN \
--input-type wp_angles_vecs_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/' \
--scenarios long_straight \
--timesteps 50000 \
--gen-expert-data \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 10 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 1 & 

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_angles_vecs_obs_info_speed_steer_ldist_light \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/expert_data_gen_longjunc/' \
--scenarios long_straight \
--timesteps 50000 \
--gen-expert-data \
--num-npc 50 \
--enable-brake \
--disable-semantic  \
--use-pid-fs \
--const-collision-penalty 3  --steer-penalty-coeff  2 \
--const-light-penalty 3 --light-penalty-speed-coeff 3 \
--collision-penalty-speed-coeff 3 \
--fs 3 --reward-norm 24 --success-reward 100 --constant-reward 1 --dqn-n-step 1 \
--lr 3e-5 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 1 & 

