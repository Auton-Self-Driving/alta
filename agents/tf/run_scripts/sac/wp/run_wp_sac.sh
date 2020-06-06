# # gpu 8
python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 0 &
sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 0 &
sleep 60
python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 0 &
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 0 &
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 0 &
sleep 60
python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 10 --code-gpu 2 --carla-gpu 2 &
sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 10 --code-gpu 2 --carla-gpu 2 &
sleep 60
python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 10 --code-gpu 2 --carla-gpu 2 &
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 10 --code-gpu 2 --carla-gpu 2 &
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 2_layer --steer-penalty-coeff 10 --code-gpu 2 --carla-gpu 2 &
sleep 60

# # gpu 8
# python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_env_runs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60