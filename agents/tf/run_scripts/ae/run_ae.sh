python run_vae.py \
--algo AE \
--mode \
--lr 5e-3 \
--epochs 100 \
--batch-size 32 \
--base-log-dir '/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/' \
--code-gpu 0 \
--data-dir '/home/scratch/tanmaya/projects/alta-logs/new_env/ppo_runs/ae_images/algo_PPO_input_wp_network_2_layer_lr_0.0002_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_finetune_vae/algo_PPO_input_wp_network_2_layer_lr_0.0002_dynamic_navigation_npc_20_brake_col_250.0_col_sp_250.0_finetune_vae_runid_2/test_images' \
--run-id 1 &