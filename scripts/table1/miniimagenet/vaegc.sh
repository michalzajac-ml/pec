NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-miniimg --model=vaegc \
        --n_epochs=1 --classes_per_task=1 --optim_scheduler=linear \
        --eval_every_n_task=1000 --resize_image_shape=32 --batch_size=32 --depth=3 \
        --reducing_layers=3 --channels=10 --z_dim=44 --fc_lay=1 --force_no_augmentations=True \
        --lr=0.003 --recon_loss=MSE
done
