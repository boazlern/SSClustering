for seed in 1 2 3 4 5; do
    python3 train.py --dataset stl10 --data_seeds $seed --n_labels 1000 --alpha 0.6 --depth 37 --us_lr 0.005 --crop_size 96 --us_rotnet_epoch --milestones 50 100 150 --s_epochs 5 --s_ema_eval --rn stl10-labels=1000,partition=$seed
done
