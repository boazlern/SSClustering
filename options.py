import argparse
import os

ARCHS = ['resnet', 'resnet34', 'resnet18', 'wide_resnet', 'resnext']
S_ALGS = ['fix_match', 'cta_clustering', 's_fixmatch', 'balanced_fixmatch', 'cta_fixmatch', 'contrastive_fixmatch',
          'mixmatch', 'uda', 'cta_uda', 'remixmatch']
US_ALGS = ['clustering', 'us_fixmatch', 'contrastive']
RESTORE_ARGS = ['arch', 'widen_factor', 'depth', 'dataset', 'sobel', 'grayscale', 'normalize_data', 'alpha']
OPTIMIZER_ARGS = ['us_lr', 's_lr', 'momentum', 'nesterov', 'us_wd', 's_wd']
TRASNFORMS_ARGS = ['color_jitter', 'h_flip', 'crop_size', 'grayscale', 'normalize_data', 'r',
                   'labeled_transform', 'unlabeled_transform']
MODEL_SPECIFICS_ARGS = ['widen_factor', 'depth', 'dropout']
SCHEDULERS = ['linear', 'cosine', 'step']
UNLABELED_TRANSFORMS = ['weak_fixmatch', 'strong_clustering', 'contrastive', 'mixmatch', 'cta', 'cta_clustering']
LABELED_TRANSFORMS = ['strong', 'rotate', 'mixmatch', 'cta']
DATASETS = ['cifar10', 'cifar100', 'svhn', 'stl10']


class epochs_func(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        new_args = []
        for value in values:
            if value.isnumeric():
                new_args.append(int(value))
            else:
                val, repetitions = value.split(':')
                new_args += [int(val)] * int(repetitions)
        setattr(namespace, option_string.strip('-'), new_args)


class BasicParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse(self):
        opts = self.parser.parse_args()
        args = vars(opts)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return opts


class TrainingOptions(BasicParser):
    def __init__(self):
        super(TrainingOptions, self).__init__()

        # paths

        self.parser.add_argument('--data_dir', default=os.path.abspath('data'), type=str, help='path to dataset')
        self.parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                                 help='path to checkpoints directory')
        self.parser.add_argument('--log_dir', default='logs', type=str,
                                 help='path to tensorboard log directory')
        self.parser.add_argument('--resume', type=str, metavar='PATH',
                                 help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--teacher_path', type=str,
                                 help='path to a model which will create K pseudo labels from each class.')
        self.parser.add_argument('--rn', default='nat_test_run', type=str,
                                 help='run name')
        self.parser.add_argument('--data_seeds', type=str,
                                 help='path to a saved model whose data seeds we are interested in')
        self.parser.add_argument('--save_pseudo', type=str,
                                 help='path to a folder for saving hard pseudo labels to use later on'
                                      'with fixmatch code in tensorflow. *** Not used anymore ***')

        # training hyper-parameters

        self.parser.add_argument('--us_lr', default=0.01, type=float,
                                 metavar='LR', help='initial learning rate for the unsupervised phase')
        self.parser.add_argument('--s_lr', default=0.03, type=float,
                                 metavar='LR', help='initial learning rate for the semi-supervised phase')
        self.parser.add_argument('--min_lr', default=0.02, type=float,
                                 metavar='LR', help='minimal lr for the semi-supervised phase in the case of '
                                                    'linear scheduler')
        self.parser.add_argument('--milestones', default=[], nargs='+', type=int,
                                 help='epochs in which to multiply lr by gamma when using step scheduler')
        self.parser.add_argument('--gamma', default=0.5, type=float,
                                 help='multiplication factor for the step scheduler. The default is 0.5')
        self.parser.add_argument('--us_scheduler', type=str, choices=SCHEDULERS, default='step',
                                 help='which scheduler to use for the unsupervised phase. default is none.')
        self.parser.add_argument('--s_scheduler', type=str, choices=SCHEDULERS, default='cosine',
                                 help='which scheduler to use for the semi-supervised phase. default is cosine.')
        self.parser.add_argument('--momentum', default=0.9, type=float,
                                 metavar='LR', help='momentum of the SGD optimizer')
        self.parser.add_argument('--nesterov', default=False, action='store_true',
                                 help='whether to use nesterov in the semi-supervised phase')
        self.parser.add_argument('--us_wd', default=1e-4, type=float,
                                 metavar='W', help='weight decay for the unsupervised phase')
        self.parser.add_argument('--s_wd', default=5e-4, type=float,
                                 metavar='W', help='weight decay for the semi-supervised phase')
        self.parser.add_argument('--dropout', default=0.0, type=float,
                                 help='dropout probability. supported only with WideResnet.')

        # flow and data

        self.parser.add_argument('--arch', '-a', metavar='ARCH', default='wide_resnet', choices=ARCHS,
                                 help='model architecture: ' + ' | '.join(ARCHS) + ' (default: wrn)')
        self.parser.add_argument('--widen_factor', default=2, type=int,
                                 help='the widen factor of the wide resnet architecture.')
        self.parser.add_argument('--depth', default=28, type=int,
                                 help='the depth of the wide resnet architecture.')
        self.parser.add_argument('--s_algo', default='cta_fixmatch', choices=S_ALGS,
                                 help='the semi-supervised algorithm to use')
        self.parser.add_argument('--us_algo', default='clustering', choices=US_ALGS,
                                 help='the unsupervised algorithm to use')
        self.parser.add_argument('--iterations', default=200, type=int, metavar='N',
                                 help='number of total alternating iterations between supervised and '
                                      'unsupervised to run. the default is 200.')
        self.parser.add_argument('--s_epochs', default=[10], nargs='+', type=str, action=epochs_func,
                                 help='number of semi-supervised epochs. It can be a different number in each'
                                      'iteration. The default is a single number - 10.')
        self.parser.add_argument('--us_epochs', default=[1], nargs='+', type=str, action=epochs_func,
                                 help='number of unsupervised epochs. It can be a different number in each'
                                      'iteration. The default is a single number - 1.')
        self.parser.add_argument('--n_batches', default=1000, type=int,
                                 help='number of batches on each supervised epoch. *** Not used anymore ***')
        self.parser.add_argument('--n_labels', type=int,
                                 help='number of total labels.')
        self.parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS,
                                 help='which dataset to run. The default is cifar-10.')
        self.parser.add_argument('--us_batch_size', default=128, type=int,
                                 help='mini-batch size in the unsupervised phase')
        self.parser.add_argument('--s_batch_size', default=64, type=int,
                                 help='mini-batch size on the labeled data')
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers (default: 4)')
        self.parser.add_argument('--r', default=3, type=int,
                                 help='number of repetitions of augmentations in clustering.')
        self.parser.add_argument('--lambda_pseudo', type=float, default=1,
                                 help='the weight of the pseudo labels loss in the semi-supervised phase (FixMatch)')
        self.parser.add_argument('--mu', default=7, type=int,
                                 help='coefficient of unlabeled batch size (FixMatch)')
        self.parser.add_argument('--confidence_threshold', default=0.95, type=float,
                                 help='confidence threshold for including unlabeled sample in loss (FixMatch).')
        self.parser.add_argument('--nat_std', default=0.0, type=float,
                                 help='the std of the nat targets.')

        # extra options

        self.parser.add_argument('--s_ema_eval', default=False, action='store_true',
                                 help='whether to use exponentially moving average of the model in ss evaluation.')
        self.parser.add_argument('--us_ema_eval', default=False, action='store_true',
                                 help='whether to use exponentially moving average of the model in unsupervised '
                                      'evaluation.')
        self.parser.add_argument('--s_ema_teacher', default=False, action='store_true',
                                 help='whether to use exponentially moving average of the model in ss phase as'
                                      'a teacher.')
        self.parser.add_argument('--us_ema_teacher', default=False, action='store_true',
                                 help='whether to use exponentially moving average of the model in unsupervised phase'
                                      'as a teacher.')
        self.parser.add_argument('--fresh_optim', default=False, action='store_true',
                                 help='when resuming, whether to retrieve optimizer state.')
        self.parser.add_argument('--us_rotnet_epoch', default=False, action='store_true',
                                 help='whether to use rotnet in unsupervised phase for full epoch every epoch.')
        self.parser.add_argument('--us_rotnet_batch', default=False, action='store_true',
                                 help='whether to use rotnet in unsupervised phase every batch.')
        self.parser.add_argument('--s_rotnet_epoch', default=False, action='store_true',
                                 help='whether to use rotnet in semi-supervised phase for full epoch every epoch.')
        self.parser.add_argument('--s_rotnet_batch', default=False, action='store_true',
                                 help='whether to use rotnet in semi-supervised phase every batch.')
        self.parser.add_argument('--rotnet_start_epochs', type=int, default=20,
                                 help='how many epochs to train only on rotnet before normal training. The default'
                                      'is 20.')
        self.parser.add_argument('--us_first', default=False, action='store_true',
                                 help='whether to start with unsupervised phase.')
        self.parser.add_argument('--ul_to_l', default=False, action='store_true',
                                 help='whether to move confident samples from unlabeled data to labeled data'
                                      'after the unsupervised phase. *** not used anymore ***')
        self.parser.add_argument('--transductive', default=False, action='store_true',
                                 help='whether to use the validation set without labels for training')
        self.parser.add_argument('--estimate_perm', default=False, action='store_true',
                                 help='whether to estimate labels permutation when going from clustering to ss.'
                                      'If not, the identity is enforced.')
        self.parser.add_argument('--estimate_freq', default=1, type=int,
                                 help='indicates the amount of iterations between each permutation estimate.')
        self.parser.add_argument('--freeze', type=int, default=0,
                                 help='the number of modules in features to freeze.')
        self.parser.add_argument('--frozen_lr', type=float, default=0.0,
                                 help='the learning rate of the almost frozen layers.')
        self.parser.add_argument('--K', default=0, type=int,
                                 help='number of most confident images from each class. *** not used anymore ***')
        self.parser.add_argument('--interleave', default=False, action='store_true',
                                 help='whether to use interleave batches as in Fixmatch.')
        self.parser.add_argument('--alpha', default=1, type=float,
                                 help='clustering alpha hyper-parameter: the ratio of the targets to initialize.'
                                      'The default is 1, i.e. having a target to each unlabeled image. ')
        self.parser.add_argument('--rho', default=0.2, type=float,
                                 help='clustering rho hyper-parameter: the distance threshold')

        self.parser.add_argument('--debug', default=False, action='store_true',
                                 help='whether to run on debug mode')
        self.parser.add_argument('--lab_gpu', type=str,
                                 help='If run on lab computers, can specify which gpu. e.g: 0,1')
        self.parser.add_argument('--comb', type=int,
                                 help='for grid search')
        self.parser.add_argument('--unsupervised_eval', type=str, default='validation',
                                 choices=['unlabeled', 'validation'],
                                 help='whether to evaluate the unsupervised epoch with the unlabeled data or the '
                                      'validation data')
        self.parser.add_argument('--only_rotnet', default=False, action='store_true',
                                 help='for ablation: in the unsupervised epoch, only rotnet will be executed.')

        # augmentations and preprocessing

        self.parser.add_argument('--sobel', default=False, action='store_true',
                                 help='whether to apply sobel filters to the image.')

        self.parser.add_argument('--color_jitter', type=float, default=0.4,
                                 help='the range of the color jitter to apply. 0 for no color jitter.')
        self.parser.add_argument('--h_flip', dest='h_flip', action='store_true',
                                 help='whether to use random horizontal flip.')
        self.parser.add_argument('--no_h_flip', dest='h_flip', action='store_false')
        self.parser.set_defaults(h_flip=True)
        self.parser.add_argument('--crop_size', type=int,
                                 help='the size of the crop.')
        self.parser.add_argument('--grayscale', default=False, action='store_true',
                                 help='whether to convert image to grayscale.')
        self.parser.add_argument('--normalize_data', default=False, action='store_true',
                                 help='whether to normalize the dataset: reduce mean and divide by std. If not, '
                                      'The images are transformed to the range [-1, 1].')
        self.parser.add_argument('--unlabeled_transform', type=str, choices=UNLABELED_TRANSFORMS,
                                 help='type of unlabeled transform. The default if not specified is like in '
                                      'the clustering and FixMatch algorithms')
        self.parser.add_argument('--labeled_transform', type=str, choices=LABELED_TRANSFORMS,
                                 help='type of labeled transform. The default if not specified is like in FixMatch')


class EvaluationOptions(BasicParser):
    def __init__(self):
        super(EvaluationOptions, self).__init__()

        # paths
        self.parser.add_argument('--ckpt', type=str, help='path to model checkpoint')
        self.parser.add_argument('--data_dir', default=os.path.abspath('data'), type=str, help='path to dataset')

        # flow and data

        self.parser.add_argument('--batch_size', default=512, type=int, help='batch size')
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers (default: 4)')
        self.parser.add_argument('--ema', default=False, action='store_true',
                                 help='whether to use exponentially moving average of the model.')

        self.parser.add_argument('--eval_data', default='validation', type=str, choices=['validation', 'labeled',
                                                                                         'unlabeled', 'all'],
                                 help='which part of the data to use for evaluation.')

        self.parser.add_argument('--only_validation', default=False, action='store_true',
                                 help='whether to use only the validation set for evaluation and not the whole'
                                      'unlabeled data.')
        self.parser.add_argument('--us', default=False, action='store_true',
                                 help='whether to evaluate the model after the clustering phase or after the '
                                      'classification phase.')
        self.parser.add_argument('--clustering_score', default=False, action='store_true',
                                 help='whether to evaluate the clustering accuracy score instead of classification '
                                      'accuracy.')
        self.parser.add_argument('--eval_train', default=False, action='store_true',
                                 help='whether to evaluate the model on the train set')

        self.parser.add_argument('--lab_gpu', type=str,
                                 help='If run on lab computers, can specify which gpu. e.g: 0,1')
        self.parser.add_argument('--K', default=5, type=int,
                                 help='number of most confident images from each class.')
        self.parser.add_argument('--top_k', default=0, type=int,
                                 help='number of permutations in the top_k evaluation')
        self.parser.add_argument('--perms_ensemble', default=False, action='store_true',
                                 help='whether to evaluate the model with ensemble of k best permutations')
        self.parser.add_argument('--runs_ensemble', default=False, action='store_true',
                                 help='whether to evaluate the model with ensemble of all runs of the same experiment.')
        self.parser.add_argument('--tolerance', default=4, type=int,
                                 help='the maximum distance of accepted permutation from the learned one.')


class CompareModelsOptions(BasicParser):
    def __init__(self):
        super(CompareModelsOptions, self).__init__()

        self.parser.add_argument('--experiments', type=str,
                                 help='ids of experiments separated by commas. e.g.: 100032,100033,100054')
        self.parser.add_argument('--exp_labels', type=str,
                                 help='names of experiments separated by commas. e.g.: lr=0.01,lr=0.02,lr=0.03')
        self.parser.add_argument('--base_folder', default='experiments', type=str,
                                 help='the folder where all experiments are saved.')
        self.parser.add_argument('--plot_mean', default=False, action='store_true',
                                 help='whether to plot the mean graph with std over runs of the same seed or all '
                                      'the graphs together.')
        self.parser.add_argument('--lowest_loss', default=False, action='store_true',
                                 help='whether to report classification/clustering accuracies according to the '
                                      'lowest loss model or the model after the last epoch. ')


class ShowGraphsOptions(BasicParser):
    def __init__(self):
        super(ShowGraphsOptions, self).__init__()

        self.parser.add_argument('--exp_id', type=str, help='id of desired experiment')
        self.parser.add_argument('--sub_exp', type=str, help='the number of run whose graphs to present')
        self.parser.add_argument('--base_folder', type=str,
                                 help='the folder where all experiments are saved.')


class MultiEvaluationOptions(EvaluationOptions, CompareModelsOptions):
    def __init__(self):
        super(MultiEvaluationOptions, self).__init__()
        self.parser.add_argument('--file_name', default='top_k-results', type=str,
                                 help='the name of the excel file with the results')
        self.parser.add_argument('--ks', default=[3], nargs='+', type=int,
                                 help='number of permutations in the top_k evaluation - can be a list.')
        self.parser.add_argument('--end_iter', default=False, action='store_true',
                                 help='whether to evaluate the model after the end of training or the one'
                                      'with the lowest loss.')

