python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 0 \
--lr 4e-4 --run-id 1 &
sleep 60
python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 1 \
--lr 4e-4 --run-id 2 &
sleep 60
python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 2 \
--lr 4e-4 --run-id 3 &
sleep 60
