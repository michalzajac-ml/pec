NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-cifar10 --model=derpp \
        --n_epochs=1 --buffer_size=500 --eval_every_n_task=1000 \
        --take_one_batch_from_buffer=True --minibatch_size=10 --batch_size=10 \
        --force_no_augmentations=True --classes_per_task=2 --optim_scheduler=none \
        --lr=0.0003 --alpha=1.0 --beta=1.0
done
