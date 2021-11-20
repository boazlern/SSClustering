for seed in 1 2 3 4 5; do # each seed was rerun 3 times. 
    python3 train.py --dataset cifar10 --data_seeds $seed --n_labels 40 --crop_size 32 --us_rotnet_epoch --only_rotnet --milestones 100 --s_ema_eval --rn ablation-fixmatch+rotnet,partition=$seed
    
    python3 train.py --dataset cifar10 --data_seeds $seed --n_labels 40 --crop_size 32 --milestones 100 --s_ema_eval --rn ablation-fixmatch+clustering,partition=$seed
done
