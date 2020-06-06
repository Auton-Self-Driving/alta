python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.3 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 4 \
--clip 0.3 \
--videos \
--run-id 2 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 4 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 5 &
sleep 60






python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.3 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 10 \
--clip 0.3 \
--videos \
--run-id 2 &
sleep 60








python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.3 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios long_straight \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 1e-4 \
--no-epochs 20 \
--clip 0.3 \
--videos \
--run-id 2 &
sleep 60


