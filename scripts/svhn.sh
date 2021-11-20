for n_labels in 10 20 30 40 100 250; do # the 10,20,30,40 labels experiments were repeated 3 times for each seed.
    for seed in 1 2 3 4 5; do
        python3 train.py --dataset svhn --data_seeds $seed --n_labels $n_labels --alpha 0.6 --no_h_flip --crop_size 32 --rotnet_start_epochs 0 --milestones 20 50 100 150 --s_epochs 5 --s_ema_eval --rn svhn-labels=$n_labels,partition=$seed  # note that as in FixMatch we don't horizontally flip images in SVHN dataset. We also don't use rotnet.
    done
done
