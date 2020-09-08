
# (Run 1 on ssh gpu18, run2 & 3 on ssh gpu19)
sleep 600
python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_64_fs_3' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_64_fs_3' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_16_32_64_64_fs_3' \
--scenarios dynamic_navigation \
--timesteps 16000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60






# Testing 

python run_code.py \
--algo PPO \
--test \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
--test-trails 2 \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--num-npc 70 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 