NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-mnist --model=pec --n_epochs=1 \
        --classes_per_task=1 --optim_scheduler=linear --eval_every_n_task=1000 \
        --force_no_augmentations=True --batch_size=1 --lr=0.01 \
        --pec_architecture=mlp --pec_activation=gelu \
        --pec_teacher_width_multiplier=500 --pec_width=10 --pec_output_dim=99
done
