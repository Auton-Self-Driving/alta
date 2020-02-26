# # gpu 8
python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_inp-wp_npc60/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
sleep 60

# # gpu 8
# python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
# python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/sac_runs_1/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --layers 1_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 &
# sleep 60
