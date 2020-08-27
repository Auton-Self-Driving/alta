# # ssh gpu11
# # Task: Straight
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios straight \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios straight \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3
# sleep 1800








# # Task: curved
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios curved \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios curved \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3
# sleep 1800









# # Task: Navigation
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3
# sleep 1800






# # Task: Dynamic Navigation
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --num-npc 15 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3
# sleep 1800








# ssh gpu3
# Task: No Crash Empty

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &
# sleep 60


# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 3 &
# sleep 60





# # Task: No Crash Regular

python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--num-npc 20 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_regular \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60


python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--city_name 'Town02' \
--num-npc 15 \
--test \
--test-trails 5 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_regular \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60






# # Task: No Crash Dense

python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--num-npc 100 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_dense \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60


python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--city_name 'Town02' \
--num-npc 70 \
--test \
--test-trails 5 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_8280000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights8280000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_dense \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60

























# ssh gpu14
# Task: Straight
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios straight \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios straight \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60








# # Task: curved
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios curved \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios curved \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2
# sleep 1800









# # Task: Navigation
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60






# # Task: Dynamic Navigation
# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-collision \
# --test \
# --test-trails 5 \
# --city_name 'Town02' \
# --num-npc 15 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2
# sleep 3600








# ssh gpu3
# Task: No Crash Empty

# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60


# python ../../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
# --input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
# --network CustomPolicy2 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --fstack 3 \
# --finetune-vae --ae-lr 1e-3 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.2 \
# --videos \
# --run-id 2 &
# sleep 60





# Task: No Crash Regular

python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--num-npc 20 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_regular \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60


python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--city_name 'Town02' \
--num-npc 15 \
--test \
--test-trails 5 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_regular \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60






# Task: No Crash Dense

python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--num-npc 100 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_dense \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60


python ../../../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--city_name 'Town02' \
--num-npc 70 \
--test \
--test-trails 5 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/ae_weights/ae_8640000' \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.001_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10__runid_2/models/ppo2_weights8640000.zip' \
--input-type wp_vae_obs_info_speed_steer_ldist_goal_light \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae/16_32_64_64/' \
--scenarios no_crash_dense \
--timesteps 8000000 \
--fstack 3 \
--finetune-vae --ae-lr 1e-3 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 2 &
sleep 60

