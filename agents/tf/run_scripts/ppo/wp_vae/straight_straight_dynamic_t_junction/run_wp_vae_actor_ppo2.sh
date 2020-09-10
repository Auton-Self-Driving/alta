python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 1 --code-gpu 1 \
--videos

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 1 --code-gpu 1 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1 \
--carla-gpu 2 --code-gpu 2 \
--videos

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1 \
--carla-gpu 2 --code-gpu 2 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 5 \
--carla-gpu 0 --code-gpu 0 \
--videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 5 \
--carla-gpu 0 --code-gpu 0 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 10 \
--carla-gpu 3 --code-gpu 3 \
--videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 10 \
--carla-gpu 3 --code-gpu 3 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 \
--videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 50 \
--carla-gpu 0 --code-gpu 0 \
--videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 50 \
--carla-gpu 0 --code-gpu 0 &

python -m cProfile -o brake_profile1.txt run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight \
--timesteps 1000000 \
--num-npc 120 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0 \
--videos &

python -m cProfile -o profile_navigation.txt run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/benchmark_navigation/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios navigation \
--timesteps 1000000 \
--carla-gpu 1 --code-gpu 1 \
--finetune-vae \
--lr 2e-4 --run-id 6 &



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0 \
--videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1 


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1 

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 2 --code-gpu 2 --videos

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 3 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 1 --code-gpu 1 --videos

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 3 --code-gpu 3


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 1 --code-gpu 1 --videos

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple1/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 3 --code-gpu 3

python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 0 \
--lr 2e-4 --run-id test1 \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 3 --code-gpu 3

GPU3:

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 --videos


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0 --videos &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0 &


--ent-coef

GPu8 
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 1 --code-gpu 1 --videos --ent-coef 0.01 &


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    python run_code.py \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --algo PPO \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --input-type wp_vae \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --network CustomPolicy2 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --scenarios straight_dynamic \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --timesteps 1000000 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --num-npc 2 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --lr 2e-4 --run-id 2 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --finetune-vae --enable-brake \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --const-collision-penalty 0 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --collision-penalty-speed-coeff 100 \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    --carla-gpu 1 --code-gpu 1  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 3 --code-gpu 3 --videos  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 3 --code-gpu 3  --ent-coef 0.01 &

gpu 10

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3 --videos  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 100 \
--carla-gpu 3 --code-gpu 3  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0 --videos  --ent-coef 0.01 &
sleep 60
python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 &


gpu11

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 3 --code-gpu 3 --videos  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 3 --code-gpu 3  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 10 \
--carla-gpu 0 --code-gpu 0 --videos  --ent-coef 0.01 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 10 \
--carla-gpu 2 --code-gpu 2 --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0 --videos  --ent-coef 0.01 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 &

gpu5

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id test2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id test99 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 3 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --videos


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --videos

gpu 9

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 1000 \
--carla-gpu 3 --code-gpu 3  --ent-coef 0.01

trying fs

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --videos --fs 2 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --fs 2

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --fs 5


gpu 14

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 1000


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 1000 --videos &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000  --fs 2 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 1000  --fs 2 &

gpu4

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000  --fs 2 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 1000  --fs 2 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000 &



python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 1000  --fs 2 &

gpu10

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000 &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000 --fs 2 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 1000 --fs 5 &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.69_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 2 --code-gpu 2 --n-steps 1000 --fs 2 &


gpu11

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 3 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 0 --code-gpu 0  --ent-coef 0.01 --n-steps 1000  --fs 2 --videos &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 4 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 1000  --fs 5 --videos &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 2000  &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 500 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 2000  &

gpu8 

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 2000  &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 250 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 2000  &


python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 1 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 750 \
--carla-gpu 2 --code-gpu 2  --ent-coef 0.01 --n-steps 2000  &

python run_code.py \
--algo PPO \
--input-type wp_vae \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/npc_st_signal_brake_simple_variance_0.3_col_fixed/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id 2 \
--finetune-vae --enable-brake \
--const-collision-penalty 0 \
--collision-penalty-speed-coeff 750 \
--carla-gpu 1 --code-gpu 1  --ent-coef 0.01 --n-steps 2000  &



python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 0 \
--lr 2e-4 --run-id test10_0_10_0_10 \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 3 --code-gpu 3 --fs 2

python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 0 \
--lr 2e-4 --run-id test10_0_10_0_10_ \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 3 --code-gpu 3 --fs 4


python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 0 \
--lr 2e-4 --run-id test10_0_10_0_10 \
--finetune-vae --enable-brake \
--const-collision-penalty 1000 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 3 --code-gpu 3 --fs 6


python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 0 \
--lr 2e-4 --run-id test_collision_1 \
--finetune-vae --enable-brake \
--const-collision-penalty 1 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 2 --code-gpu 2

python run_code.py \
--algo PID_TUNE \
--input-type wp \
--network CustomPolicy2 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/pid_test/' \
--vae_model_path '/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json' \
--scenarios straight_dynamic \
--timesteps 1000000 \
--num-npc 2 \
--lr 2e-4 --run-id test_collision_npc_1 \
--finetune-vae --enable-brake \
--const-collision-penalty 1 \
--collision-penalty-speed-coeff 0 \
--carla-gpu 2 --code-gpu 2
