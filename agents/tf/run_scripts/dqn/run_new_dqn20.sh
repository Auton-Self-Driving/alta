python run_code.py \
--algo DQN \
--input-type wp_speed_goal \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_dqn_runs/t_junction_reduced_actions_2/' \
--scenarios t_junction \
--timesteps 2000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs \
--const-collision-penalty 0 --steer-penalty-coeff  2 \
--collision-penalty-speed-coeff 0 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--videos \
--carla-gpu 2 --code-gpu 2 \
--run-id 1 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed_goal \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_dqn_runs/t_junction_reduced_actions_2/' \
--scenarios t_junction \
--timesteps 2000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs \
--const-collision-penalty 0 --steer-penalty-coeff  2 \
--collision-penalty-speed-coeff 0 \
--fs 1 \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 2 &

sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed_goal \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/new_dqn_runs/t_junction_reduced_actions_2/' \
--scenarios t_junction \
--timesteps 2000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic  \
--use-pid-fs \
--const-collision-penalty 0 --steer-penalty-coeff  2 \
--collision-penalty-speed-coeff 0 \
--fs 1  --full-tb-log \
--lr 1e-6 --buffer-size 1000000 \
--carla-gpu 2 --code-gpu 2 \
--run-id 3 &