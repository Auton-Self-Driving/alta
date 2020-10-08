# Differences: Lane Invasion, Collision Penalty, Brake
#ssh gpu18 (1, 2, 3), gpu19 (4)
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
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

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
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

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 3 &
sleep 60

#ssh gpu19
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60









#ssh gpu21

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 1 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 2 &
sleep 60

#ssh gpu19
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 3 &
sleep 60

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy3 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/neurips_new_vae/updated' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model_pretrained_nohazard.json' \
--finetune-vae \
--disable-lane-invasion-termination \
--disable-traffic-light \
--disable-obstacle-info \
--scenarios navigation \
--timesteps 2000000 \
--const-collision-penalty 100 --collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 \
--lr 2e-4 \
--val-interval 20000 \
--videos \
--run-id 4 &
sleep 60