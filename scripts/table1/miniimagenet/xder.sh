NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-miniimg --model=xder --n_epochs=1 \
        --buffer_size=500 --eval_every_n_task=1000 \
        --take_one_batch_from_buffer=True --minibatch_size=10 --batch_size=10 \
        --simclr_batch_size=10 --classes_per_task=1 --optim_scheduler=none \
        --lr=0.001 --alpha=0.6 --beta=0.9 --lambd=0.1 --eta=0.01 --m=0.7
done
