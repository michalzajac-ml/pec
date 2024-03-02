NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-mnist --model=vaegc --n_epochs=1 \
        --classes_per_task=1 --optim_scheduler=none --eval_every_n_task=1000 \
        --batch_size=32 --depth=0 --fc_lay=3 --fc_units=5 --z_dim=10 \
        --force_no_augmentations=True --lr=0.003 --recon_loss=BCE
done
