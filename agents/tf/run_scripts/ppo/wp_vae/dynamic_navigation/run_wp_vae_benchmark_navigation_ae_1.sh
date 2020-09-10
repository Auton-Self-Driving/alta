python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_sac_ae_runs/' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--videos \
--run-id 1 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_sac_ae_runs/' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--run-id 2 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_sac_ae_runs/' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 \
--run-id 3 &