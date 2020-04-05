python ../run_code.py --algo PPO --task "Ant-v2" --run-id run1 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco/" --timesteps 5000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 3 --code-gpu 3 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60
python ../run_code.py --algo PPO --task "Ant-v2" --run-id run2 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco/" --timesteps 5000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 3 --code-gpu 3 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60
python ../run_code.py --algo PPO --task "Ant-v2" --run-id run3 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco/" --timesteps 5000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 3 --code-gpu 3 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60
python ../run_code.py --algo PPO --task "Ant-v2" --run-id run4 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco/" --timesteps 5000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 3 --code-gpu 3 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60
python ../run_code.py --algo PPO --task "Ant-v2" --run-id run5 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco/" --timesteps 5000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 3 --code-gpu 3 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60
'''python ../run_code.py --algo PPO --run-id run6 --base-log-dir "/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_corr-term/" --timesteps 1000000 --lr 2e-4 --network 2_layer --steer-penalty-coeff 0 --carla-gpu 2 --code-gpu 2 --const-collision-penalty 0 --collision-penalty-speed-coeff 0 --input-type wp --ent-coef 0.005 &
sleep 60'''
