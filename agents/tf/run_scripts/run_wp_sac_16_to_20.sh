python ../run_code.py --algo SAC --run-id run1 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 64 &
sleep 60
python ../run_code.py --algo SAC --run-id run2 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 64 &
sleep 60
python ../run_code.py --algo SAC --run-id run4 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 64 &
sleep 60
python ../run_code.py --algo SAC --run-id run5 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 64 &
sleep 60
python ../run_code.py --algo SAC --run-id run6 --base-log-dir '/zfsauton2/home/mayankgu/alta/alta-logs/sac/' --timesteps 1000000 --lr 3e-4 --buffer-size 200000 --network 2_layer --steer-penalty-coeff 0 --code-gpu 1 --carla-gpu 1 --videos --const-collision-penalty 10 --collision-penalty-speed-coeff 3 --input-type wp --batch-size 64 &
sleep 60
