NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-cifar10 --model=labels_trick \
        --n_epochs=1 --eval_every_n_task=1000 --classes_per_task=1 \
        --force_no_augmentations=True --optim_scheduler=none --lr=0.03 \
        --batch_size=1
done
