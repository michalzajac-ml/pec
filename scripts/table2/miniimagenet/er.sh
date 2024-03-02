NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-miniimg --model=er --n_epochs=1 \
        --buffer_size=500 --eval_every_n_task=1000 --minibatch_size=1 \
        --batch_size=1 --classes_per_task=5 --optim_scheduler=none --lr=0.01
done
