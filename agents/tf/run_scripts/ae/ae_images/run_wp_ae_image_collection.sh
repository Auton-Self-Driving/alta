python run_vae.py \
--algo AE \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_data' \
--num-npc 60 \
--timesteps 1000000 \
--carla-gpu 0 --code-gpu 0 \
--run-id 1



python run_code.py \
--algo PPO \
--input-type wp \
--network 2_layer \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_data' \
--vae_model_path '/media/hdd/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--num-npc 20 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--videos \
--lr 2e-4 --run-id 5 &
sleep 60



python run_code.py \
--algo PPO \
--input-type wp \
--network 2_layer \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_data' \
--vae_model_path '/media/hdd/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--num-npc 60 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60


python run_code.py \
--algo PPO \
--input-type wp \
--network 2_layer \
--base-log-dir '/media/hdd/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/AE_data' \
--vae_model_path '/media/hdd/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--num-npc 90 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--carla-gpu 0 --code-gpu 0 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60