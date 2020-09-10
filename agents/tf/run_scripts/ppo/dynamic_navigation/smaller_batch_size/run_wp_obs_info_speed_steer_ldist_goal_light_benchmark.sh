python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60

# Test trial
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_3/models/ppo2_weights5880000.pkl' \
--test \
--test-trails 2 \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 4 &
sleep 60






python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 4 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 5 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 6 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 110 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 7 &
sleep 60







python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 4 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 5 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 6 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/long_straight_framestack/wp_obs_info_speed_steer_ldist_goal_light_fixed_scen/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000_runid_1/models/ppo2_weights4480000.pkl' \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 150 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 1000 \
--lr 2e-4 \
--no-epochs 10 \
--clip 0.1 \
--videos \
--run-id 7 &
sleep 60