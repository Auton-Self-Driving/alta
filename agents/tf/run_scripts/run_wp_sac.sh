# # gpu 8
#python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 2000000 --lr 3e-4 --buffer-size 50000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp_vae
#sleep 60
#python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 2000000 --lr 3e-4 --buffer-size 50000 --network 3_layer --steer-penalty-coeff 0 --code-gpu 2 --carla-gpu 2 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp_vae
#sleep 60
#python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 2000000 --lr 3e-4 --buffer-size 50000 --network 3_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 3 --collision-penalty-speed-coeff 5 --input-type wp_vae --batch-size 512
#sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
sleep 60
python ../run_code.py --algo SAC --run-id run6 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
sleep 60
python ../run_code.py --algo SAC --run-id run7 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
sleep 60

#python ../run_code.py --algo SAC --run-id run8 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 20 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
#sleep 6
#python ../run_code.py --algo SAC --run-id run9 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 20 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
#sleep 6
#python ../run_code.py --algo SAC --run-id run10 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 20 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
#sleep 6
#python ../run_code.py --algo SAC --run-id run11 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 20 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
#sleep 6
#python ../run_code.py --algo SAC --run-id run12 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 20 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 1024 &
#sleep 6


#python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 50000 --network 3_layer --steer-penalty-coeff 0 --code-gpu 0 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp_vae
#sleep 60
#python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60
#python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60
#python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60
#python ../run_code.py --algo SAC --run-id run3 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60
#python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60
#python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac_inp-wp_npc60/' --timesteps 1000000 --lr 3e-4 --buffer-size 1000000 --network 2_layer --steer-penalty-coeff 10 --code-gpu 3 --carla-gpu 3 --videos --const-collision-penalty 100 --collision-penalty-speed-coeff 100 --input-type wp
#sleep 60

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
