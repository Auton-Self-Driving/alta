# Differences: Lane Invasion, Collision Penalty, Brake
#ssh gpu19
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion \
--enable-static \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion \
--enable-static \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 3 &
sleep 60







# Task: Dynamic navigation (fs:1)
# ssh gpu21
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_1' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 3 &
sleep 60








# Task: Dynamic navigation (fs:3)
# run1 on ssh gpu18, run2 & 3 on ssh gpu15
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--fstack 3 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--fstack 3 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 5 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_16_fs_3' \
--finetune-vae \
--disable-lane-invasion \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 70 \
--fstack 3 \
--timesteps 4000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 6 &
sleep 60