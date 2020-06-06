sleep 300
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
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
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
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
--fs 3 \
--fstack 3 \
--train-vae --ae-lr 5e-4 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4  --num-npc 20 \
--run-id 3 &



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae_runid_3/ae_weights/ae_1000000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae_runid_3/ppo2_measurements_weights1000000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--finetune-vae --ae-lr 5e-4 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4  --num-npc 20 \
--run-id 4 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae_runid_3/ae_weights/ae_1000000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_sac_ae_runs/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae/algo_PPO_input_wp_vae_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_fs_3_fstack_3_train_vae_runid_3/ppo2_measurements_weights1000000.pkl' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--fs 3 \
--fstack 3 \
--finetune-vae --ae-lr 5e-4 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4  --num-npc 20 \
--run-id 5 &
sleep 60