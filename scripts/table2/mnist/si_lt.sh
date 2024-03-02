NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-mnist --model=si_lt --n_epochs=1 \
        --eval_every_n_task=1000 --classes_per_task=2 --optim_scheduler=linear \
        --lr=0.003 --batch_size=1 --force_no_augmentations=True --c=0.5 --xi=1.0
done
