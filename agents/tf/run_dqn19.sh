python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 100000 \
--videos \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 100000 \
--carla-gpu 1 --code-gpu 1 \
--full-tb-log \
--run-id 2 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 100000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 3 &

sleep 120
sleep 240
python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 1000000 \
--videos \
--carla-gpu 3 --code-gpu 3 \
--run-id 1 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--full-tb-log \
--run-id 2 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 1000000 \
--carla-gpu 3 --code-gpu 3 \
--run-id 3 &

sleep 120
python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 500000 \
--videos \
--carla-gpu 1 --code-gpu 1 \
--run-id 1 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 500000 \
--carla-gpu 1 --code-gpu 1 \
--full-tb-log \
--run-id 2 &
sleep 120

python run_code.py \
--algo DQN \
--input-type wp_speed \
--base-log-dir '/zfsauton2/home/hiteshar/research/alta-logs/dqn_runs/t-junction_3/' \
--scenarios t_junction \
--timesteps 1000000 \
--num-npc 0 \
--enable-brake \
--enable-static \
--disable-semantic \
--use-pid-fs \
--const-collision-penalty 250 \
--collision-penalty-speed-coeff 250 \
--fs 3 \
--lr 3e-4 --buffer-size 500000 \
--carla-gpu 1 --code-gpu 1 \
--run-id 3 &