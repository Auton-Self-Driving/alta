python run_code.py \
--algo PPO \
--input-type wp_obs_dist \
--network 2_layer \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env/ppo_runs/wp_obs_npc_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 2 --code-gpu 2 --fs 5 --n-steps 1000 \
--lr 2e-4 --run-id 1 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_dist \
--network 2_layer \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env/ppo_runs/wp_obs_npc_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 2 --code-gpu 2 --fs 5 --n-steps 1000 \
--lr 2e-4 --run-id 2 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_dist \
--network 2_layer \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env/ppo_runs/wp_obs_npc_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 3 --code-gpu 3 --fs 10 --n-steps 1000 \
--lr 2e-4 --run-id 1 --videos &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_obs_dist \
--network 2_layer \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env/ppo_runs/wp_obs_npc_variance_0.1/' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 500 \
--collision-penalty-speed-coeff 500 \
--vae_model_path '/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json' \
--carla-gpu 3 --code-gpu 3 --fs 10 --n-steps 1000 \
--lr 2e-4 --run-id 2 &
sleep 60