python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 1000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_16' \
--vae_model_path '/home/scratch/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_models/old_data_npc/fstack3//net_16_32_64_16/algo_AE_lr_0.005_batchsize_128_fs_3_epochs_50_runid_2/ae_weights/ae_14300' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 1000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_32' \
--vae_model_path '/home/scratch/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_models/old_data_npc/fstack3//net_16_32_64_32/algo_AE_lr_0.005_batchsize_128_fs_3_epochs_50_runid_1/ae_weights/ae_21800' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--carla-gpu 0 --code-gpu 0 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60





python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--scenarios long_straight \
--timesteps 1000000 \
--enable-brake \
--num-npc 70 \
--fstack 3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/net_16_32_64_64' \
--vae_model_path '/home/scratch/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_models/old_data_npc/fstack3//net_16_32_64_64/algo_AE_lr_0.005_batchsize_128_fs_3_epochs_50_runid_1/ae_weights/ae_18100' \
--finetune-vae \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--const-light-penalty 100 --light-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60