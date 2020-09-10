python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ae_weights/ae_680000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_1/ppo2_measurements_weights680000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60









python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 2 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 2 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ae_weights/ae_510000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_2/ppo2_measurements_weights510000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60








python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ae_weights/ae_610000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_3/ppo2_measurements_weights610000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60






python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 4 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios straight \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 4 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 4 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios curved \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 4 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 4 \
--city_name Town01 \
--test --test-trails 2 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ae_weights/ae_890000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation_reproduce/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_navigation_finetune_vae_runid_4/ppo2_measurements_weights890000.pkl' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 4 \
--city_name Town02 \
--test --test-trails 2 --videos &
sleep 60