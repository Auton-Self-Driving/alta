python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_straight1_s20/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 60 \
--carla-gpu 0 --code-gpu 1 \
--lr 2e-4 --run-id 1 \
--video --finetune-vae &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_straight1_s20/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 60 \
--carla-gpu 0 --code-gpu 2 \
--lr 2e-4 --run-id 2 \
--video --finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_straight1_s20/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 3 \
--lr 2e-4 --run-id 1 \
--video --finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 60 \
--carla-gpu 1 --code-gpu 2 \
--lr 2e-4 --run-id 1 \
--video --finetune-vae &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 60 \
--carla-gpu 0 --code-gpu 1 \
--lr 2e-4 --run-id 2 \
--video --finetune-vae


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 1 --code-gpu 3 \
--lr 2e-4 --run-id 1 \
--video --finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 2 \
--lr 2e-4 --run-id 2 \
--video --finetune-vae


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 1 --code-gpu 0 \
--lr 4e-4 --run-id 1 \
--finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 1 \
--lr 4e-4 --run-id 2 \
--finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 3 \
--lr 8e-4 --run-id 1 \
--finetune-vae


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 3 \
--lr 1e-5 --run-id 1 \
--finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 3 \
--lr 1e-5 --run-id 1 \
--finetune-vae

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 2 \
--lr 1e-3 --run-id 1 \
--finetune-vae


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_tjunction/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 120 \
--carla-gpu 0 --code-gpu 1 \
--lr 5e-5 --run-id 1 \
--finetune-vae


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id test2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 0\
--carla-gpu 1 --code-gpu 1 \
--videos 