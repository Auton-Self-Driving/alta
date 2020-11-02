echo $LIBS

python run_vae.py \
    --algo AE \
    --data-dir '../../../alta-logs/' \
    --base-log-dir '../../../alta-logs/' \
    --run-id 0
