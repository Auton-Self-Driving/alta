python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--test \
--test-trails 2 \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__mb_10__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000/algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__mb_10__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_runid_1/models/ppo2_weights5520000.zip' \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 1 &







python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60






sleep 300
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
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
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
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
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60







sleep 600
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60






sleep 900
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 2 --code-gpu 2 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60






sleep 300
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60







sleep 600
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 1 --code-gpu 1 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.1 \
--videos \
--run-id 3 &
sleep 60






sleep 900
python run_code.py \
--algo PPO \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--num-npc 70 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
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
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/updated_monotonic/wp_obs_info_speed_steer_ldist_goal_light' \
--carla-gpu 3 --code-gpu 3 \
--n-steps 20000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 20 \
--clip 0.2 \
--videos \
--run-id 3 &
sleep 60



