python run_vae.py --algo VAE --run-id 1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/vae_1/' --timesteps 2000000 --lr 5e-3 --num-npc 60 --vae-zsize 64 --code-gpu 1 --carla-gpu 0 &
sleep 60
python run_vae.py --algo VAE --run-id 1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/vae_1/' --timesteps 2000000 --lr 5e-3 --num-npc 60 --vae-zsize 128 --code-gpu 2 --carla-gpu 0 &
sleep 60
python run_vae.py --algo VAE --run-id 1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/vae_1/' --timesteps 2000000 --lr 5e-3 --num-npc 60 --vae-zsize 256 --code-gpu 3 --carla-gpu 0 &
