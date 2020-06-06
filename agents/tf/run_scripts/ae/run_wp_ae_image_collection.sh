python run_code.py \
--algo PPO \
--input-type wp \
--network 2_layer \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_old.json' \
--scenarios dynamic_navigation \
--num-npc 20 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--carla-gpu 2 --code-gpu 2 \
--videos \
--lr 2e-4 --run-id 3 &
sleep 60