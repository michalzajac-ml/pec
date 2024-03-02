NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-cifar100 --model=ewc_on \
        --n_epochs=1 --eval_every_n_task=1000 --gamma=1.0 \
        --force_no_augmentations=True --classes_per_task=1 \
        --optim_scheduler=linear --lr=0.03 --batch_size=10 --e_lambda=10.0
done
