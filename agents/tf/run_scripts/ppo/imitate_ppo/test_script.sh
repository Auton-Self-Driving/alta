python ../../../run_code.py \
--algo PPO \
--disable-lane-invasion-termination \
--test-trails 1 \
--num-npc 15 \
--disable-sample-npc \
--light-thresold 15 \
--min-light-thresold 6 \
--agent_model_path '/home/vkadi/expert_model/ppo2_weights6000000.zip' \
--input-type wp_obs_info_speed_steer_ldist_goal_light \
--network 2_layer \
--scenarios dynamic_navigation \
--timesteps 8000000 \
--const-collision-penalty 250 --collision-penalty-speed-coeff 250 \
--const-light-penalty 250 --light-penalty-speed-coeff 250 \
--base-log-dir '/home/vkadi/ResnetWP_alta/logs/Imitation/' \
--carla-gpu 0 --code-gpu 0 \
--n-steps 10000 \
--lr 2e-4 \
--no-epochs 10 \
--no-minibatches 10 \
--clip 0.1 \
--run-id 1 \
--disable-semantic \
--city_name Town02 \
--imitate \
#--test \
#--dataset-path '/zfsauton2/home/vkadi/projects/alta/agents/tf/run_scripts/ppo/imitate_ppo/imitation_data_combined1.p'
#--disable-semantic \
#--test \

#--dataset-path '/zfsauton2/home/vkadi/projects/alta/agents/tf/run_scripts/ppo/imitate_ppo/imitation_data_combined1.p'
#