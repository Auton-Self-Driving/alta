python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 0 --code-gpu 0 --network 1_layer --lr 2e-4 --ent-coef 0.005 --run-id 1 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 0 --code-gpu 0 --network 1_layer --lr 2e-4 --ent-coef 0.005 --run-id 2 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 1 --code-gpu 1 --network 1_layer --lr 2e-4 --ent-coef 0.005 --run-id 3 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 1 --code-gpu 1 --network 1_layer --lr 2e-4 --ent-coef 0.005 --run-id 4 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 1 --code-gpu 1 --network 1_layer --lr 2e-4 --ent-coef 0.005 --run-id 5 &
sleep 60

python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 2 --code-gpu 2 --network 2_layer --lr 2e-4 --ent-coef 0.005 --run-id 1 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 2 --code-gpu 2 --network 2_layer --lr 2e-4 --ent-coef 0.005 --run-id 2 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 2 --code-gpu 2 --network 2_layer --lr 2e-4 --ent-coef 0.005 --run-id 3 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 3 --code-gpu 3 --network 2_layer --lr 2e-4 --ent-coef 0.005 --run-id 4 &
sleep 60
python run_code.py --algo PPO --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/' --timesteps 1000000 --carla-gpu 3 --code-gpu 3 --network 2_layer --lr 2e-4 --ent-coef 0.005 --run-id 5 &
sleep 60