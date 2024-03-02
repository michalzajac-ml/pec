NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-svhn --model=ewc_on --n_epochs=1 \
        --eval_every_n_task=1000 --gamma=1.0 --classes_per_task=1 \
        --optim_scheduler=none --lr=0.001 --batch_size=10 --e_lambda=100.0 \
        --force_no_augmentations=True --balance_truncate_data=True
done