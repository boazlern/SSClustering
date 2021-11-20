for seed in 1 2 3 4 5; do
    python3 train.py --s_algo mixmatch --dataset cifar10 --data_seeds $seed --n_labels 40 --crop_size 32 --us_rotnet_epoch --milestones 50 100 250 --s_ema_eval --unlabeled_transform mixmatch --labeled_transform mixmatch --mu 1 --s_epochs 1 --iterations 300 --interleave --rn mixmatch-partition=$seed
done
