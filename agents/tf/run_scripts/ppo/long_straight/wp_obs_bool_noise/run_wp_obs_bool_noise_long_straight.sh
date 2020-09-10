python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--videos \
--noise-dim 1 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--noise-dim 1 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--noise-dim 1 \
--run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--noise-dim 4 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--noise-dim 4 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--noise-dim 4 \
--run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--videos \
--noise-dim 16 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--noise-dim 16 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--noise-dim 16 \
--run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--noise-dim 64 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--noise-dim 64 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--noise-dim 64 \
--run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--videos \
--noise-dim 128 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--noise-dim 128 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool_noise \
--network 2_layer \
--scenarios long_straight \
--timesteps 4000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool_noise' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--noise-dim 128 \
--run-id 3 &
sleep 60