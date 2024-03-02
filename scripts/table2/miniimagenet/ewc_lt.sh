NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-miniimg --model=ewc_lt \
        --n_epochs=1 --eval_every_n_task=1000 --force_no_augmentations=True \
        --gamma=1.0 --classes_per_task=5 --optim_scheduler=none --lr=0.003 \
        --batch_size=10 --e_lambda=1.0
done
