NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-svhn --model=si_lt --n_epochs=1 \
        --eval_every_n_task=1000 --classes_per_task=1 --optim_scheduler=none \
        --lr=0.01 --batch_size=32 --force_no_augmentations=True \
        --balance_truncate_data=True --c=1.0 --xi=1.0
done
