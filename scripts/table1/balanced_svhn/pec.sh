NUM_SEEDS="${1:-1}"
echo "Will run with $NUM_SEEDS seeds."
for ((SEED = 1; SEED <= $NUM_SEEDS; SEED++)); do
    echo "Running with seed $SEED..."
    python3 main.py --seed="$SEED" --dataset=seq-svhn --model=pec --n_epochs=1 \
        --classes_per_task=1 --optim_scheduler=linear --eval_every_n_task=1000 \
        --force_no_augmentations=True --batch_size=1 --lr=0.0001 \
        --pec_architecture=cnn --pec_conv_layers="(60, 3, 1)" --pec_output_dim=743 \
        --pec_conv_reduce_spatial_to=5 --balance_truncate_data=True
done
