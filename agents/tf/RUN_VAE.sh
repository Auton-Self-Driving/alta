echo $LIBS

python run_vae.py \
    --algo VAE \
    --data-dir '../../../alta-logs/' \
    --base-log-dir '../../../alta-logs/' \
    --run-id 1
