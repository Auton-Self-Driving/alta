
# ssh gpu18
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 1 &

# Got killed
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &





# ssh gpu18
# (Run 1 got killed)
sleep 750
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 1 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &






# ssh gpu19
# (run3 got killed)
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-5 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 1 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-5 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_cnn_obs_info_speed_steer_ldist_goal_light \
--binarized-image \
--single-channel-image \
--network CNN \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_jtn_images/cnn/nature_64/' \
--scenarios long_straight_junction \
--timesteps 16000000 \
--fstack 3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--num-npc 70 \
--n-steps 10000 \
--lr 6e-5 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &