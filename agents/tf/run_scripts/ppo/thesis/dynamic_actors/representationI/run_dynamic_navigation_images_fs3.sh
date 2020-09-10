# Runs without obstacle info
# ssh gpu19(1), gpu19 (2&3)
python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_ldist_goal_light \
--network CustomPolicy4 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_6520000' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_ldist_goal_light \
--network CustomPolicy4 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_6520000' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_speed_steer_ldist_goal_light \
--network CustomPolicy4 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_6520000' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--run-id 3 &
sleep 60