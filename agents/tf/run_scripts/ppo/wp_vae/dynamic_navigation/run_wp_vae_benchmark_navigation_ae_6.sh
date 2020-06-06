python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 1 \
--fstack 3 \
--train-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4  --num-npc 20 \
--videos \
--run-id 1 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 1 \
--fstack 3 \
--train-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4  --num-npc 20 \
--run-id 2 &

sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 1 \
--fstack 3 \
--train-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4  --num-npc 20 \
--run-id 3 &

sleep 60