for seed in 1 2 3 4 5; do
    python3 train.py --s_algo uda --dataset cifar10 --data_seeds $seed --n_labels 40 --crop_size 32 --us_rotnet_epoch --milestones 20 50 100 --s_ema_eval --labeled_transform mixmatch --mu 1 --iterations 120 --confidence_threshold 0.8 --rn uda-partition=$seed
done
