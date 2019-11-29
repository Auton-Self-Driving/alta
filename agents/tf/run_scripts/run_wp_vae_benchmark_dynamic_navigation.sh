python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 2 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 2 &
sleep 60







python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 2 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_wo_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_wo_term/' \
--vae_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ae_ppo_wp_vae_benchmark_navigation.json' \
--agent_model_path '/home/tanmaya/projects/alta/agents/tf/trained_models/ppo_wp_vae_benchmark_navigation.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 2 &
sleep 60