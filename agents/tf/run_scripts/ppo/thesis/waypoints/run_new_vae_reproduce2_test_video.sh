
# Navigation
python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion \
--disable-traffic-light-termination \
--disable-collision \
--test \
--test-trails 1 \
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
--num-npc 100 \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60