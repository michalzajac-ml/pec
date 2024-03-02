NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-svhn --model=derpp --n_epochs=1 \
        --buffer_size=500 --eval_every_n_task=1000 \
        --take_one_batch_from_buffer=True --minibatch_size=10 --batch_size=10 \
        --classes_per_task=2 --optim_scheduler=linear --lr=0.0003 --alpha=0.5 \
        --beta=1.0 --force_no_augmentations=False --balance_truncate_data=True
done
