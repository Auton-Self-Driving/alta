python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 1000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 2000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 1000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 2000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 1000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 2000 \
--videos \
--lr 7e-5 --run-id 1 &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 1000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 2000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 1000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 2000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 1000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 2000 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60




python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 1000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 1 --n-steps 2000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 1000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 3 --n-steps 2000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 1000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_braking_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 1 --code-gpu 1 --fs 10 --n-steps 2000 \
--videos \
--lr 6e-4 --run-id 1 &
sleep 60