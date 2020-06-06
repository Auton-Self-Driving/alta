python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--validation \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--validation \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--validation \
--run-id 3 &
sleep 60




python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--videos \
--validation \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--videos \
--validation \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_bool \
--network 2_layer \
--scenarios long_straight \
--timesteps 2000000 \
--num-npc 70 \
--enable-brake \
--const-collision-penalty 500 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_bool' \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--videos \
--validation \
--run-id 3 &
sleep 60