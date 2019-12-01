python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae_runid_3/ae_weights/ae_150000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae_runid_3/ppo2_measurements_weights150000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 3 \
--city_name Town02 \
--test --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae_runid_4/ae_weights/ae_90000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_finetune_vae_runid_4/ppo2_measurements_weights90000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--num-npc 20 \
--lr 2e-4 --run-id 4 \
--city_name Town02 \
--test --videos &
sleep 60




python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae_runid_1/ae_weights/ae_360000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae_runid_1/ppo2_measurements_weights360000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--num-npc 20 \
--lr 2e-4 --run-id 1 \
--city_name Town02 \
--test --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae_runid_2/ae_weights/ae_330000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_wo_penalty_wo_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_disable_collision__enable_static_1000__finetune_vae_runid_2/ppo2_measurements_weights330000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--disable-collision \
--enable-static \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--num-npc 20 \
--lr 2e-4 --run-id 2 \
--city_name Town02 \
--test --videos &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae_runid_1/ae_weights/ae_550000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae_runid_1/ppo2_measurements_weights550000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 1 \
--city_name Town02 \
--test --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/' \
--vae_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae_runid_2/ae_weights/ae_500000' \
--agent_model_path '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_reproduce_w_penalty_w_term/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_dynamic_navigation_pretrained_agent__npc_20_col_250.0_col_sp_250.0_finetune_vae_runid_2/ppo2_measurements_weights500000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--finetune-vae \
--carla-gpu 1 --code-gpu 1 \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--num-npc 20 \
--lr 2e-4 --run-id 2 \
--city_name Town02 \
--test --videos &
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