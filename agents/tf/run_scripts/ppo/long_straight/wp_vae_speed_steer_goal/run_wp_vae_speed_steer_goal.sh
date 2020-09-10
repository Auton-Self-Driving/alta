python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 0 --code-gpu 0 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 --run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 3 &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 --run-id 3 &
sleep 60










python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 1 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_1' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 --run-id 3 &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 3 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_goal \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 2000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_32_fs_3' \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 --run-id 3 &
sleep 60