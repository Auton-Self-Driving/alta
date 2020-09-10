python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce_w_collision/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_old.json' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 1 &
sleep 60

python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce_w_collision/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_old.json' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 2 &
sleep 60

python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce_w_collision/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_old.json' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 3 &
sleep 60

python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce_w_collision/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_old.json' \
--scenarios navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 4 &
sleep 60