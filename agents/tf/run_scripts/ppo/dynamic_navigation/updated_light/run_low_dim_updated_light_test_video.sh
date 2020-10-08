# # ssh gpu9
# # Task: Straight
# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios straight \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 18000
















# # ssh gpu9
# # Task: Curved
# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios curved \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 18000












# # ssh gpu9
# # Task: Navigation
# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 18000

















# #ssh gpu14
# # Task: Dynamic Navigation
# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios dynamic_navigation \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60






# Task: No Crash Empty

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_empty \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3
# sleep 1800 





# Task: No Crash Regular

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --num-npc 20 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2
# sleep 1800

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 15 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_regular \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3
# sleep 1800







# Task: No Crash Dense

python ../../../../run_code.py \
--algo PPO \
--disable-lane-invasion-termination \
--disable-traffic-light-termination \
--test \
--test-trails 1 \
--num-npc 100 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios no_crash_dense \
--timesteps 8000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --num-npc 100 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_dense \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --test \
# --test-trails 5 \
# --num-npc 100 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_dense \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 70 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_1/models/ppo2_weights6000000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_dense \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 3 --code-gpu 3 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 1 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 70 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_2/models/ppo2_weights5720000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_dense \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 2 &
# sleep 60

# python ../../../../run_code.py \
# --algo PPO \
# --disable-lane-invasion-termination \
# --disable-traffic-light-termination \
# --city_name 'Town02' \
# --num-npc 70 \
# --test \
# --test-trails 5 \
# --disable-sample-npc \
# --light-thresold 15 \
# --min-light-thresold 6 \
# --agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10_/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_epochs_10__clip_0.1__mb_10__runid_3/models/ppo2_weights6480000.zip' \
# --input-type wp_obs_info_speed_steer_ldist_goal_light \
# --network 2_layer \
# --scenarios no_crash_dense \
# --timesteps 8000000 \
# --const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
# --const-light-penalty 250 --light-penalty-speed-coeff 250 \
# --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_light/wp_obs_info_speed_steer_ldist_goal_light' \
# --carla-gpu 2 --code-gpu 2 \
# --n-steps 10000 \
# --lr 2e-4 \
# --no-epochs 10 \
# --no-minibatches 10 \
# --clip 0.1 \
# --videos \
# --run-id 3 &
# sleep 60