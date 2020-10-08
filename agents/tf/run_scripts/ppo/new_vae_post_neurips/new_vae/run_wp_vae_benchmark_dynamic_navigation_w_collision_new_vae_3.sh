python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_w_collision_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--num-npc 20 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--videos \
--lr 2e-4 --run-id 1 &
sleep 60

python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_w_collision_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--num-npc 20 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 2 &
sleep 60

python ../run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_dynamic_navigation_w_collision_new_vae/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--scenarios dynamic_navigation \
--timesteps 1000000 \
--enable-brake \
--finetune-vae \
--num-npc 20 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--lr 2e-4 --run-id 3 &
sleep 60
