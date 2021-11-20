for n_labels in 10 20 30 40 100 250; do  # the 10,20,30,40 labels experiments were repeated 3 times for each seed.
    for seed in 1 2 3 4 5; do
        python3 train.py --dataset cifar10 --data_seeds $seed --n_labels $n_labels --crop_size 32 --us_rotnet_epoch --milestones 100 --s_ema_eval --rn cifar10-labels=$n_labels,partition=$seed
    done
done
