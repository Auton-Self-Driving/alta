# ssh gpu15
# Straight
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

# Curved
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1
sleep 1800


# Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60


# Dynamic Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--num-npc 15 \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1
sleep 1800




# Run 2

# Straight
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

# Curved
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2
sleep 1800

# ssh gpu18
# Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60


# Dynamic Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--num-npc 15 \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2
sleep 1800





# Run 4

# Straight
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios straight \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

# Curved
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios curved \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4
sleep 1800


# Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60


# Dynamic Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--num-npc 15 \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios dynamic_navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4








# NoCrash Benchmarking
# ssh gpu15
# no_crash_empty
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

# no_crash_regular
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 15 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1
sleep 1800


# no_crash_dense
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 100 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/models/ppo2_weights1080000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_1/ae_weights/ae_1080000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 70 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60


# Run 2

# no_crash_empty
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

# no_crash_regular
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 15 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

# ssh gpu18
# no_crash_dense
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 100 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/models/ppo2_weights920000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_2/ae_weights/ae_920000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 70 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60





# Run 4

# no_crash_empty
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4
sleep 1800

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_empty \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

# no_crash_regular
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 20 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_regular \
--num-npc 15 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60


# no_crash_dense
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 100 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

python ../../../run_code.py \
--algo PPO \
--city_name 'Town02' \
--disable-lane-invasion \
--disable-traffic-light-termination \
--test \
--test-trails 5 \
--disable-sample-npc \
--agent_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/models/ppo2_weights1540000.zip' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4_/algo_PPO_input_wp_vae_network_CustomPolicy3_lr_0.0002_navigation_disable_light__disable_obs__col_250.0_col_sp_250.0_finetune_vae_epochs_4__clip_0.2__mb_4__runid_4/ae_weights/ae_1540000' \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--finetune-vae \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios no_crash_dense \
--num-npc 70 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60

