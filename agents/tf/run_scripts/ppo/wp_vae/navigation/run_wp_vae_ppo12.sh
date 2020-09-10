python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--finetune-vae \
--videos \
--lr 4e-4 --run-id 1 &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--finetune-vae \
--lr 4e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--finetune-vae \
--lr 4e-4 --run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--finetune-vae \
--lr 4e-4 --run-id 4 &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--finetune-vae \
--lr 4e-4 --run-id 5 &




python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 1 \
--finetune-vae \
--videos \
--lr 5e-4 --run-id 1 &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 2 \
--finetune-vae \
--lr 5e-4 --run-id 2 &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 3 \
--finetune-vae \
--lr 5e-4 --run-id 3 &
sleep 60









python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/pylon5/cc5fpsp/tanmaya/alta-logs/new_env/ppo_runs/benchmark_curved/' \
--vae_model_path '/pylon5/cc5fpsp/tanmaya/alta/agents/tf/trained_models/ae_model.json' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--finetune-vae \
--lr 5e-4 --run-id 1 &
sleep 60

/pylon5/cc5fpsp/tanmaya/alta/agents/tf