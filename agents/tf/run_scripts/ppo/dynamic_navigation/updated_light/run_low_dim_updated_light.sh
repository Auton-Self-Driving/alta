#gpu3

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60




#gpu4

sleep 300
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
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
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60





#ssh gpu14

sleep 600
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60




#ssh gpu15

sleep 900
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

# ssh gpu18
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60